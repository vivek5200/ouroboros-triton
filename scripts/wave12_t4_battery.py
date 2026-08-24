"""wave12_t4_battery.py — importable study battery behind t4_full_suite.sh.

Two self-contained entry points (stdlib + torch only):

* :func:`perm_study` — the permutation-aware ``d_head=8`` study. Builds a
  random dense attention problem (no Triton kernel required) and shows BOTH
  halves of the documented RoPE-layout bridge
  (:func:`src.kernels.reference_attention.perm` /
  :func:`~src.kernels.reference_attention.apply_channel_perm`):

  - comparing the half-split-layout computation against the interleaved
    golden reference WITHOUT the channel permutation produces a large
    max_diff (pure layout mismatch, O(1));
  - bridging the golden output through ``apply_channel_perm`` collapses the
    same comparison to floating-point noise (rotation commutes with the
    relabeling).

  At ``d_head == 8`` this matters doubly: ``perm`` has ORDER 3 there (cycles
  ``(1 4 2)(3 5 6)``), so blind round-trips through the forward map do NOT
  restore the original — the study also quantifies that trap and confirms
  the explicit inverse is exact.

* :func:`head_sweep` — the placement-head scaling sweep. Drives the core
  repo's harness (``ouroboros-core/src/scaling_study.py``) EXACTLY as
  published: ``StudyConfig(d_model, epochs, instances, ...)`` cells ->
  ``scaling_study(configs)`` rows -> ``render_table(rows)`` ASCII table.
  Rows carry the contract metric ``lift = post - fresh``; the suite fails
  if ANY lift is non-finite.

Import contract (important): the two repos both ship a top-level package
named ``src``, so this module NEVER imports either at module load time.
Each study function inserts ITS repo root at ``sys.path[0]`` immediately
before its own lazy import. Run the two studies in separate interpreter
processes (t4_full_suite.sh does) so the cached ``src`` cannot collide.

Paths: the sibling repos are located relative to THIS file by default
(<this repo>/../ouroboros-core). Override with :func:`set_core_src` /
:func:`set_triton_root` (absolute repo-root paths) or the
``OUROBOROS_PARENT`` environment variable.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

__all__ = [
    "DEFAULT_T4_SWEEP",
    "TINY_SMOKE_CONFIGS",
    "perm_study",
    "head_sweep",
    "render_table",
    "set_core_src",
    "set_triton_root",
]

# Placement-head sweep grid from the Wave-12 T4 plan: (d_model, epochs,
# instances). Maps onto scaling_study.StudyConfig field-for-field; batch_size
# keeps the harness default (8).
DEFAULT_T4_SWEEP = (
    (16, 6, 128),
    (32, 8, 256),
    (64, 10, 384),
)

# CPU-smoke preset for hosts without a GPU (and CI): same pipeline, seconds
# of runtime, still real learning per the harness docstring.
TINY_SMOKE_CONFIGS = (
    (16, 1, 6),
    (32, 1, 8),
)

_CORE_SRC_OVERRIDE: str | None = None
_TRITON_ROOT_OVERRIDE: str | None = None


# ---------------------------------------------------------------------------
# Path plumbing
# ---------------------------------------------------------------------------


def _parent_dir() -> str:
    """Directory that contains the four side-by-side Ouroboros repos."""
    env = os.environ.get("OUROBOROS_PARENT")
    if env:
        return os.path.abspath(env)
    # <parent>/ouroboros-triton/scripts/wave12_t4_battery.py -> parent
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def set_core_src(core_repo_root: str | None) -> None:
    """Point :func:`head_sweep` at the ouroboros-core REPO ROOT explicitly."""
    global _CORE_SRC_OVERRIDE
    _CORE_SRC_OVERRIDE = os.path.abspath(core_repo_root) if core_repo_root else None


def set_triton_root(triton_repo_root: str | None) -> None:
    """Point :func:`perm_study` at the ouroboros-triton REPO ROOT explicitly."""
    global _TRITON_ROOT_OVERRIDE
    _TRITON_ROOT_OVERRIDE = (
        os.path.abspath(triton_repo_root) if triton_repo_root else None
    )


def _triton_root() -> str:
    if _TRITON_ROOT_OVERRIDE:
        return _TRITON_ROOT_OVERRIDE
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)  # this file lives in <triton repo>/scripts/


def _core_root() -> str:
    if _CORE_SRC_OVERRIDE:
        return _CORE_SRC_OVERRIDE
    candidate = os.path.join(_parent_dir(), "ouroboros-core")
    if os.path.isdir(candidate):
        return candidate
    raise RuntimeError(
        "cannot locate ouroboros-core next to "
        f"{_parent_dir()!r}; call set_core_src(<repo root>) or set OUROBOROS_PARENT"
    )


def _push_sys_path_front(path: str) -> None:
    path = os.path.abspath(path)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - T4 always has torch
        raise RuntimeError(
            "wave12_t4_battery requires torch (stdlib+torch budget); "
            "install torch first"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Section C — permutation-aware d_head=8 study (no kernel needed)
# ---------------------------------------------------------------------------


def _softmax(xs: list[float]) -> list[float]:
    peak = max(xs)
    exps = [math.exp(x - peak) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def _halfsplit_rope(vector, position: int, half: int, base: float) -> list[float]:
    """The KERNEL-side RoPE convention: pair ``j`` is ``(x[j], x[j+half])``.

    Mirrors the layout documented in reference_attention's module docstring
    (what the Triton kernel rotates in memory), as opposed to the oracle's
    interleaved pairing handled by ``reference_attention.rope_rotate``.
    """
    d_head = 2 * half
    out = [0.0] * d_head
    for j in range(half):
        angle = position * base ** (-2.0 * j / d_head)
        c, s = math.cos(angle), math.sin(angle)
        x0, x1 = float(vector[j]), float(vector[j + half])
        out[j] = x0 * c - x1 * s
        out[j + half] = x0 * s + x1 * c
    return out


def _dense_attention(
    qs, ks, vs, rope_fn, scale: float
) -> list[list[float]]:
    """Causal-free dense attention rows: softmax(q·k/scale) @ V."""
    n = len(qs)
    out_rows = []
    for i in range(n):
        qr = rope_fn(qs[i], i)
        scores = []
        for j in range(n):
            kr = rope_fn(ks[j], j)
            scores.append(sum(a * b for a, b in zip(qr, kr)) * scale)
        weights = _softmax(scores)
        out_rows.append([
            sum(weights[j] * vs[j][c] for j in range(n)) for c in range(len(vs[0]))
        ])
    return out_rows


def perm_study(
    d_head: int = 8, n_tokens: int = 12, base: float = 10000.0, seed: int = 7
) -> dict:
    """Prove the channel-permutation bridge on a random dense problem.

    Returns a dict with the two headline numbers (``unpermuted_max_diff`` —
    comparing layouts naively — and ``permuted_max_diff`` — after routing the
    golden output through ``apply_channel_perm``), the permutation map, the
    forward-vs-inverse round-trip diagnostics at this ``d_head``, and an
    overall ``ok`` flag (``permuted << unpermuted``).
    """
    if d_head % 2 != 0 or d_head < 2:
        raise ValueError(f"d_head must be even and >= 2, got {d_head}")
    half = d_head // 2

    _push_sys_path_front(_triton_root())
    from src.kernels.reference_attention import (
        apply_channel_perm,
        invert_channel_perm,
        perm,
        rope_rotate,
    )

    torch = _require_torch()
    generator = torch.Generator().manual_seed(seed)
    qs = torch.randn(n_tokens, d_head, generator=generator).tolist()
    ks = torch.randn(n_tokens, d_head, generator=generator).tolist()
    vs = torch.randn(n_tokens, d_head, generator=generator).tolist()

    scale = 1.0 / math.sqrt(d_head)

    # Golden side: interleaved-pair RoPE (the audited oracle primitive).
    golden = _dense_attention(qs, ks, vs, lambda vec, pos: rope_rotate(vec, pos, base), scale)

    # Kernel side: SAME math, half-split memory layout — payloads relabeled
    # through apply_channel_perm, rotation in the (j, j+half) pairing.
    def halfsplit(vec, pos):
        return _halfsplit_rope(vec, pos, half, base)

    sim = _dense_attention(
        [apply_channel_perm(row, half) for row in qs],
        [apply_channel_perm(row, half) for row in ks],
        [apply_channel_perm(row, half) for row in vs],
        halfsplit,
        scale,
    )

    # Naive comparison (WRONG at d_head > 4): treat layouts as identical.
    unpermuted_max_diff = max(
        abs(sim[i][c] - golden[i][c])
        for i in range(n_tokens)
        for c in range(d_head)
    )

    # Bridged comparison: relabel the golden output into the kernel's layout.
    golden_bridged = [apply_channel_perm(row, half) for row in golden]
    permuted_max_diff = max(
        abs(sim[i][c] - golden_bridged[i][c])
        for i in range(n_tokens)
        for c in range(d_head)
    )

    # Round-trip diagnostics: at d_head=8 the forward map has order 3, so
    # apply(apply(x)) != x, while the EXPLICIT inverse restores x exactly.
    probe = torch.randn(1, d_head, generator=generator).tolist()[0]
    once = apply_channel_perm(probe, half)
    fwd_roundtrip_err = max(
        abs(a - b) for a, b in zip(apply_channel_perm(once, half), probe)
    )
    inv_roundtrip_err = max(
        abs(a - b) for a, b in zip(invert_channel_perm(once, half), probe)
    )
    perm_map = [perm(t, half) for t in range(d_head)]

    ok = (permuted_max_diff < 1e-6 < unpermuted_max_diff) and inv_roundtrip_err == 0.0
    return {
        "d_head": d_head,
        "half": half,
        "n_tokens": n_tokens,
        "rope_base": base,
        "seed": seed,
        "perm_map": perm_map,
        "perm_is_involution": perm_map == [
            perm(perm(t, half), half) for t in range(d_head)
        ],
        "unpermuted_max_diff": unpermuted_max_diff,
        "permuted_max_diff": permuted_max_diff,
        "ratio": (unpermuted_max_diff / permuted_max_diff)
        if permuted_max_diff > 0.0
        else float("inf"),
        "forward_roundtrip_max_err": fwd_roundtrip_err,
        "inverse_roundtrip_max_err": inv_roundtrip_err,
        "ok": ok,
        "verdict": (
            "PASS: permuted max_diff << un-permuted max_diff"
            if ok
            else "FAIL: permutation bridge did not collapse the diff"
        ),
    }


# ---------------------------------------------------------------------------
# Section D — placement-head sweep through ouroboros-core scaling_study
# ---------------------------------------------------------------------------


def head_sweep(configs=None) -> list[dict]:
    """Run the placement-head scaling grid via the core harness.

    ``configs`` defaults to :data:`DEFAULT_T4_SWEEP`; each triple is
    ``(d_model, epochs, instances)`` mapped onto
    ``scaling_study.StudyConfig`` verbatim (other fields keep the published
    defaults). Returns the harness' own result rows (contract keys
    ``d_model/epochs/instances/fresh/post/lift``), each augmented with
    ``_seconds``/``_device`` diagnostics that ``render_table`` ignores.
    """
    triples = tuple(DEFAULT_T4_SWEEP if configs is None else configs)
    _push_sys_path_front(_core_root())
    from src.scaling_study import StudyConfig, scaling_study  # exact API

    torch = _require_torch()
    cuda_ok = bool(torch.cuda.is_available())  # informational only; the
    device_name = (                           # harness owns all compute.
        torch.cuda.get_device_name(0) if cuda_ok else "cpu"
    )

    study_configs = [
        StudyConfig(d_model=d_model, epochs=epochs, instances=instances)
        for d_model, epochs, instances in triples
    ]
    rows = []
    for cfg, triple in zip(study_configs, triples):
        started = time.perf_counter()
        row = scaling_study([cfg])[0]
        row["_seconds"] = round(time.perf_counter() - started, 3)
        row["_device"] = device_name
        row["_triple"] = triple
        rows.append(row)
    return rows


def render_table(rows: list[dict]) -> str:
    """Delegate to the core harness' published ``render_table``."""
    _push_sys_path_front(_core_root())
    from src.scaling_study import render_table as core_render_table

    return core_render_table(rows)


# ---------------------------------------------------------------------------
# CLI (convenience; t4_full_suite.sh drives these via import instead)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = (
        "usage: wave12_t4_battery.py perm-study [--d-head N]\n"
        "       wave12_t4_battery.py head-sweep [--tiny]\n"
    )
    if not argv:
        print(usage, end="", flush=True)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "perm-study":
        d_head = 8
        if "--d-head" in rest:
            d_head = int(rest[rest.index("--d-head") + 1])
        result = perm_study(d_head=d_head)
        import json

        print(json.dumps(result, indent=2))
        print(result["verdict"])
        return 0 if result["ok"] else 1
    if command == "head-sweep":
        configs = TINY_SMOKE_CONFIGS if "--tiny" in rest else None
        rows = head_sweep(configs=configs)
        print(render_table(rows))
        finite = all(math.isfinite(float(r["lift"])) for r in rows)
        for r in rows:
            print(
                f"d_model={r['d_model']} epochs={r['epochs']} "
                f"instances={r['instances']} lift={r['lift']:.4f} "
                f"finite={math.isfinite(float(r['lift']))}"
            )
        return 0 if finite else 1
    print(usage, end="", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
