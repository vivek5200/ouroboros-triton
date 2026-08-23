#!/usr/bin/env python3
"""Module 6 §8 end-to-end HA demonstration: Supervisor-Worker serving session.

One realistic session against :class:`src.serving.Supervisor`:

  * 3 stateless workers join; ``w1`` is PRE-ARMED (``fail_next=True``) to
    simulate a Triton-kernel segfault mid-stream.
  * A stream of 12 requests of varying sizes (mixed ASCII/multibyte, so the
    documented "length = UTF-8 BYTE length" rule matters) is routed through
    ``Supervisor.submit()`` — some targeted at a chosen worker, some left to
    the supervisor's first-alive routing.
  * The armed worker takes request #7 (index 6), segfaults mid-flight:
    ``submit`` degrades to the Phase-1 Safe Queue (documented sentinel),
    frees the in-flight chain FIRST (zero block leak), blast radius = one
    request.
  * The stream continues on the surviving workers; one live chain grows via
    supervisor-side ``[EXPAND]``.
  * Post-stream: ``health_pass`` flags the corpse, ``restart`` respawns it
    with zero state migration, ``drain_safe_queue`` re-dispatches the queued
    request FIFO through ``submit()`` itself onto a FRESH chain.
  * Teardown releases every chain; a consistency report compares the block
    table and KV-cache against the ledger.

The process exits 0 iff ALL invariants hold::

    python3 scripts/demo_serving.py            # full narration + PASS/FAIL
    python3 scripts/demo_serving.py --quiet    # exit code only

Programmatic use: ``run_demo(verbose=False) -> dict`` (importable core; the
test suite drives exactly this entry point).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` importable whether this runs as a script (python3 puts the
# SCRIPT's dir on sys.path, not the repo root) or via importlib from tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.block_table import BLOCK_SIZE  # noqa: E402
from src.serving import SAFE_QUEUE_PREFIX, Supervisor  # noqa: E402

MAX_BLOCKS = 256
TOTAL_REQUESTS = 12

#: Session workload: 12 requests of deliberately varying sizes. Byte lengths
#: span 1..~250 B => 1..4 blocks each; two entries are multibyte so their
#: UTF-8 BYTE length differs from their character count (documented choice).
REQUESTS = [
    "ping",                                #   4 B -> 1 block
    "x" * 64,                              #  64 B -> 1 block (exact fit)
    "y" * 65,                              #  65 B -> 2 blocks (spill by 1)
    "z" * 128,                             # 128 B -> 2 blocks (exact fit)
    "héllo wörld",                         # 13 ch / 15 B -> 1 block (multibyte)
    "b" * 200,                             # 200 B -> 4 blocks
    "CRASH-ME: the armed worker dies on this one",  # 45 B -> 1 block
    "c" * 96,                              #  96 B -> 2 blocks
    "δ" * 80,                              # 80 ch / 160 B -> 3 blocks (multibyte)
    "?",                                   #   1 B -> 1 block (minimum chain)
    "d" * 250,                             # 250 B -> 4 blocks
    "e" * 64,                              #  64 B -> 1 block
]

#: Routing plan, aligned with REQUESTS: a worker id targets that worker;
#: None lets the supervisor route to the first alive worker. ``w1`` (armed)
#: is deliberately first scheduled at index 6 => the crash hits MID-STREAM,
#: after six healthy requests. Later slots never name ``w1`` while it is
#: down; post-restart traffic would find it alive again.
ROUTING_PLAN = [
    "w0", "w2", "w0", "w2", "w0", "w2",   # healthy warm-up across w0/w2
    "w1",                                  # index 6: ARMED -> segfault here
    "w0", "w2",                            # stream survives the blast radius
    None, None, None,                      # supervisor first-alive routing
]


def _kv_snapshot_ok(sup: Supervisor) -> tuple[bool, int, int]:
    """Check KV-cache entries match live chains (peak-time audit).

    For every owned request_id: the recorded head walks cleanly, the walked
    length times BLOCK_SIZE equals the tracked capacity, every physical
    block is allocated, and no physical block is shared by two chains or
    sitting on the free list.

    Returns:
        ``(consistent, n_chains, total_live_blocks)``.
    """
    seen: set[int] = set()
    free_ids = set(sup.block_table.free_list)
    total = 0
    try:
        for rid, head in sup.kv_cache.items():
            row = sup.block_table.page_table_row(head)
            if len(row) * BLOCK_SIZE != sup.chain_capacity[rid]:
                return False, len(sup.kv_cache), total
            for bid in row:
                if (
                    bid in seen
                    or bid in free_ids
                    or sup.block_table.blocks[bid] is None
                ):
                    return False, len(sup.kv_cache), total
                seen.add(bid)
            total += len(row)
    except RuntimeError:
        return False, len(sup.kv_cache), total  # broken/unwalkable chain
    return True, len(sup.kv_cache), total


def run_demo(verbose: bool = True) -> dict:
    """Run the full HA session and return the invariant report dict.

    Args:
        verbose: emit the per-request narration and the final PASS/FAIL
            report (the CLI prints; the test suite stays silent).

    Returns:
        Dict of invariant fields; ``all_pass`` is True iff every invariant
        holds (this is what the CLI exit code mirrors).
    """
    log = print if verbose else (lambda *a, **k: None)

    sup = Supervisor(max_blocks=MAX_BLOCKS)
    for wid in ("w0", "w1", "w2"):
        sup.add_worker(wid, fail_next=(wid == "w1"))  # w1 PRE-ARMED to crash

    log("=" * 74)
    log(" Module 6 §8 HA Serving Demo — Supervisor owns table+KV; workers die")
    log(f" workers: w0, w1 (fail_next ARMED), w2 | block_table: {MAX_BLOCKS} x {BLOCK_SIZE}t")
    log("=" * 74)

    num_free_start = sup.block_table.num_free
    blocks_allocated = 0
    blocks_freed = 0
    served_first_pass: list[str] = []
    queued_requests: list[str] = []
    drained_served: list[str] = []
    live_chains: dict[int, str] = {}  # request_id -> request (held chains)
    crash_zero_leak = True
    crash_event: dict = {}
    expanded: dict = {}

    for i, (request, target) in enumerate(zip(REQUESTS, ROUTING_PLAN)):
        payload_blocks = Supervisor._payload_blocks(request)
        free_before = sup.block_table.num_free
        result = sup.submit(request, worker_id=target)
        rid = sup.last_request_id
        blocks_allocated += payload_blocks
        if result.startswith(SAFE_QUEUE_PREFIX):
            # Blast radius contained: submit already freed the in-flight
            # chain BEFORE queueing, so num_free must be back to pre-call.
            freed_now = free_before - sup.block_table.num_free + payload_blocks
            blocks_freed += payload_blocks  # ledger: crash released its chain
            crash_zero_leak &= sup.block_table.num_free == free_before
            queued_requests.append(request)
            dead = [w for w, wk in sup.workers.items() if not wk.alive]
            crash_event = {
                "index": i, "worker": dead[0] if dead else "?",
                "freed_blocks": freed_now,
            }
            log(
                f"[{i + 1:02d}/{TOTAL_REQUESTS}] -> {target}: "
                f"{len(request.encode())} B ({payload_blocks} blk) "
                f"*** WorkerCrash *** degraded to Safe Queue "
                f"(chain freed first: {freed_now} blk back)"
            )
        else:
            live_chains[rid] = request
            served_first_pass.append(request)
            log(
                f"[{i + 1:02d}/{TOTAL_REQUESTS}] -> {target or 'auto'}: "
                f"{len(request.encode('utf-8'))} B ({payload_blocks} blk) SERVED"
            )
        # Mid-session growth: [EXPAND] one live chain (+200 B => 4 blocks),
        # exercising supervisor-side expansion bookkeeping.
        if i == 8 and live_chains:
            grow_rid = min(live_chains)  # oldest still-held chain
            tail = sup.expand_request(grow_rid, extra_bytes=200)
            blocks_allocated += 4  # ceil(200 / 64)
            expanded = {"rid": grow_rid, "tail": tail}
            log(
                f"           [EXPAND] request #{grow_rid} grew by 4 blocks "
                f"(new tail={tail}, capacity now "
                f"{sup.chain_capacity[grow_rid]} t)"
            )

    # ---- Phase 2: detect, restart, drain --------------------------------
    dead_workers = sup.health_pass()
    for wid in dead_workers:
        sup.restart(wid)
    log("-" * 74)
    log(f"health_pass: dead={dead_workers} -> restart (zero state migration)")

    drain_results = sup.drain_safe_queue()
    for request, result in zip(queued_requests, drain_results):
        if result.startswith(SAFE_QUEUE_PREFIX):
            continue  # would fail drain_exact_once below; never expected here
        rid = sup.last_request_id
        blocks_allocated += Supervisor._payload_blocks(request)
        live_chains[rid] = request
        drained_served.append(request)
        log(
            f"drain: re-dispatched queued request on FRESH chain "
            f"({Supervisor._payload_blocks(request)} blk) -> SERVED"
        )
    safe_queue_empty = sup.safe_queue == []

    # ---- Peak-time KV-cache / block-table audit -------------------------
    kv_ok_peak, n_chains, live_blocks = _kv_snapshot_ok(sup)

    # ---- Teardown: release every held chain ------------------------------
    for rid in sorted(live_chains):
        head = sup.kv_cache[rid]
        blocks_freed += sup.block_table.chain_len(head)
        sup.release(rid)

    num_free_end = sup.block_table.num_free
    num_free_delta = num_free_end - num_free_start

    # ---- Invariants -------------------------------------------------------
    all_final = served_first_pass + drained_served
    served_exactly_once = (
        sorted(all_final) == sorted(REQUESTS)  # every request, once, no dups
        and len(all_final) == len(set(all_final))
    )
    drain_exact_once = (
        drained_served == queued_requests  # FIFO, each exactly once
        and len(drained_served) == len(set(drained_served))
        and safe_queue_empty
        and not any(r.startswith(SAFE_QUEUE_PREFIX) for r in drain_results)
    )
    kv_consistent = (
        kv_ok_peak
        and sup.kv_cache == {}
        and sup.chain_capacity == {}
        and num_free_delta == 0
    )
    leak_free = (
        crash_zero_leak
        and num_free_delta == 0
        and blocks_allocated == blocks_freed
    )

    checks: list[tuple[str, bool, str]] = [
        (
            "served_exactly_once",
            served_exactly_once,
            f"{len(all_final)}/{TOTAL_REQUESTS} requests served exactly once "
            f"({len(served_first_pass)} first-pass + {len(drained_served)} drained)",
        ),
        (
            "safe_queue_degraded",
            len(queued_requests) > 0 and bool(crash_event),
            f"forced mid-stream crash on '{crash_event.get('worker')}' at "
            f"request #{(crash_event.get('index', -1)) + 1}: "
            f"{len(queued_requests)} request(s) safely queued, blast radius = 1",
        ),
        (
            "drain_exact_once",
            drain_exact_once,
            f"FIFO drain re-served {len(drained_served)}/{len(queued_requests)} "
            f"queued request(s) exactly once on fresh chains; queue empty="
            f"{safe_queue_empty}",
        ),
        (
            "crash_zero_leak",
            crash_zero_leak,
            "in-flight chain freed BEFORE safe-queueing (num_free restored)",
        ),
        (
            "leak_free",
            leak_free,
            f"allocated={blocks_allocated} == freed={blocks_freed}; "
            f"num_free {num_free_start} -> {num_free_end} (delta {num_free_delta})",
        ),
        (
            "kv_cache_consistent",
            kv_consistent,
            f"peak audit: {n_chains}/{n_chains} live chains matched capacity "
            f"({live_blocks} blocks, no sharing/no free overlap); cache empty "
            f"after release",
        ),
    ]
    all_pass = all(ok for _, ok, _ in checks)

    if verbose:
        log("-" * 74)
        log(f"{'Consistency Report':^74}")
        log("-" * 74)
        rows = [
            ("requests_submitted", TOTAL_REQUESTS),
            ("served_first_pass", len(served_first_pass)),
            ("safe_queued", len(queued_requests)),
            ("drained_and_served", len(drained_served)),
            ("restarted_workers", ", ".join(dead_workers) or "-"),
            ("blocks_allocated", blocks_allocated),
            ("blocks_freed", blocks_freed),
            ("num_free start/end", f"{num_free_start}/{num_free_end}"),
            ("num_free_delta", num_free_delta),
            ("kv_entries_final", len(sup.kv_cache)),
        ]
        width = max(len(k) for k, _ in rows)
        for key, val in rows:
            log(f"  {key:<{width}}  {val}")
        log("-" * 74)
        for name, ok, detail in checks:
            log(f"[{'PASS' if ok else 'FAIL'}] {name:<20}: {detail}")
        log("-" * 74)
        log(f"RESULT: {'PASS' if all_pass else 'FAIL'}")

    return {
        "total_requests": TOTAL_REQUESTS,
        "served_count": len(all_final),
        "served_direct_count": len(served_first_pass),
        "queued_count": len(queued_requests),
        "drained_count": len(drained_served),
        "safe_queue_empty": safe_queue_empty,
        "restarted_workers": list(dead_workers),
        "blocks_allocated": blocks_allocated,
        "blocks_freed": blocks_freed,
        "num_free_start": num_free_start,
        "num_free_end": num_free_end,
        "num_free_delta": num_free_delta,
        "crash_zero_leak": crash_zero_leak,
        "leak_free": leak_free,
        "drain_exact_once": drain_exact_once,
        "served_exactly_once": served_exactly_once,
        "kv_consistent": kv_consistent,
        "all_pass": all_pass,
    }


def main(argv: list[str]) -> int:
    verbose = "--quiet" not in argv and "-q" not in argv
    result = run_demo(verbose=verbose)
    if verbose:
        print(
            f"\nexit={'0' if result['all_pass'] else '1'} "
            f"(all_pass={result['all_pass']})"
        )
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
