"""Tests for Module 6 §8: Supervisor-Worker HA serving layer.

Paper contract under test:
  * Supervisor owns THE KV-cache and THE block table.
  * Workers are stateless: a Triton segfault (WorkerCrash) destroys one
    request only; the supervisor respawns a fresh worker with zero state
    migration and the lost request degrades to the Phase-1 Safe Queue.
"""

import pytest

from src.block_table import BlockTable
from src.serving import SAFE_QUEUE_PREFIX, Supervisor, Worker, WorkerCrash


def test_worker_process_is_deterministic_transform():
    w = Worker(worker_id="w1")
    w.start()
    assert w.alive is True
    assert w.process("abc") == "ok:cba"
    assert w.process("abc") == "ok:cba"  # same input -> same output


def test_process_pure_no_instance_mutation_on_success():
    """Statelessness enforced: a successful process() touches no worker state."""
    w = Worker(worker_id="w1")
    w.start()
    before = dict(vars(w))
    w.process("request-payload")
    assert vars(w) == before  # process() wrote nothing into the worker


def test_dead_worker_raises_worker_crash():
    w = Worker(worker_id="w1")
    w.start()
    w.kill()  # simulated Triton segfault
    assert w.alive is False
    with pytest.raises(WorkerCrash):
        w.process("anything")


def test_fail_next_is_one_shot_crash_on_demand():
    w = Worker(worker_id="w1", fail_next=True)
    w.start()
    with pytest.raises(WorkerCrash):
        w.process("boom")  # simulated mid-request segfault
    assert w.alive is False  # segfault kills the worker instantly


# ---------------------------------------------------------------------------
# Supervisor: ownership + routing
# ---------------------------------------------------------------------------


def make_supervisor(*worker_ids):
    sup = Supervisor()
    for wid in worker_ids:
        sup.add_worker(wid)
    return sup


def test_supervisor_owns_shared_block_table_and_kv_cache():
    sup = make_supervisor("w1")
    assert isinstance(sup.block_table, BlockTable)
    assert sup.kv_cache == {}


def test_workers_have_no_direct_table_or_cache_handles():
    """Workers reach supervisor-owned state ONLY via supervisor methods."""
    sup = make_supervisor("w1")
    for w in sup.workers.values():
        assert not hasattr(w, "block_table")
        assert not hasattr(w, "kv_cache")


def test_table_op_passthrough_hits_the_one_shared_table():
    sup = make_supervisor("w1")
    head = sup.table_op(lambda bt: bt.allocate_chain(2))
    assert sup.table_op(lambda bt: bt.chain_len(head)) == 2
    assert sup.table_op(lambda bt: bt) is sup.block_table  # same object


def test_happy_path_submit_transforms_via_alive_worker():
    sup = make_supervisor("w1", "w2")
    assert sup.submit("hello", worker_id="w1") == "ok:olleh"


def test_submit_routes_to_first_alive_worker():
    sup2 = make_supervisor("a", "b")
    sup2.workers["a"].kill()
    assert sup2.submit("xy") == "ok:yx"  # landed on 'b', not dead 'a'


def test_duplicate_worker_id_raises():
    sup = make_supervisor("w1")
    with pytest.raises(RuntimeError):
        sup.add_worker("w1")


def test_unknown_worker_id_and_restart_raise():
    sup = make_supervisor("w1")
    with pytest.raises(RuntimeError):
        sup.submit("x", worker_id="nope")
    with pytest.raises(RuntimeError):
        sup.restart("ghost")


# ---------------------------------------------------------------------------
# Crash handling: blast radius = one request
# ---------------------------------------------------------------------------


def test_crash_marks_worker_dead_and_queues_request():
    sup = Supervisor()
    sup.add_worker("w1", fail_next=True)
    result = sup.submit("lost-req")
    assert result.startswith(SAFE_QUEUE_PREFIX)  # documented sentinel choice
    assert sup.safe_queue == ["lost-req"]
    assert sup.workers["w1"].alive is False


def test_health_pass_reports_only_dead_workers():
    sup = make_supervisor("w1", "w2")
    assert sup.health_pass() == []
    sup.workers["w2"].kill()
    assert sup.health_pass() == ["w2"]
    sup.workers["w1"].kill()
    assert sup.health_pass() == ["w1", "w2"]


def test_restart_replaces_worker_and_routing_resumes_stateless():
    sup = Supervisor()
    sup.add_worker("w1", fail_next=True)
    sup.submit("r1")  # crashes w1
    old = sup.workers["w1"]
    sup.restart("w1")
    fresh = sup.workers["w1"]
    assert fresh is not old and fresh.alive is True
    # Fresh worker needs NO prior state to serve identically:
    assert fresh.process("abc") == "ok:cba"
    assert sup.submit("abc", worker_id="w1") == "ok:cba"


def test_drain_redispatches_through_submit_allocating_fresh_chains():
    """Drain re-serves FIFO via submit(): fresh chains, worker transform applied."""
    sup = Supervisor()
    sup.add_worker("w1", fail_next=True)
    sup.add_worker("w2", fail_next=True)
    assert sup.submit("first").startswith(SAFE_QUEUE_PREFIX)   # kills w1
    assert sup.submit("second").startswith(SAFE_QUEUE_PREFIX)  # kills w2
    sup.restart("w1")
    sup.restart("w2")
    free_before = sup.block_table.num_free
    results = sup.drain_safe_queue()
    assert results == ["ok:tsrif", "ok:dnoces"]  # FIFO order, real transforms
    assert sup.safe_queue == []
    assert sup.drain_safe_queue() == []  # exactly once: now empty
    # Re-dispatch went THROUGH submit(): fresh chains allocated per request.
    assert sup.last_request_id == 4  # 2 crashed submits + 2 re-dispatched
    assert len(sup.kv_cache) == 2  # crashed chains were freed, these are live
    assert sup.block_table.num_free == free_before - 2  # 1 block per 5-byte req


def test_drain_without_alive_workers_requeues_and_reports_degraded():
    """No alive worker at drain time: request is re-queued, never lost."""
    sup = Supervisor()
    sup.add_worker("w1", fail_next=True)
    sup.submit("solo")  # kills w1, safe-queues "solo"
    results = sup.drain_safe_queue()
    assert results == [SAFE_QUEUE_PREFIX + "solo"]  # degraded, not dropped
    assert sup.safe_queue == ["solo"]  # waiting for a live worker
    sup.restart("w1")
    assert sup.drain_safe_queue() == ["ok:olos"]  # served after restart


def test_kv_cache_survives_worker_death():
    sup = make_supervisor("w1")
    sup.add_worker("w2", fail_next=True)
    head = sup.table_op(lambda bt: bt.allocate_chain(1))
    res = sup.submit("persist-me", worker_id="w1")
    assert res == "ok:em-tsisrep"
    sup.kv_cache["persist-me"] = {"head": head}
    sup.workers["w1"].kill()  # node-level would die; here only worker dies
    sup.restart("w1")
    # Supervisor-owned state untouched by worker lifecycle:
    assert sup.kv_cache["persist-me"] == {"head": head}
    assert sup.table_op(lambda bt: bt.chain_len(head)) == 1


# ---------------------------------------------------------------------------
# Resource semantics: chains sized by payload, crash frees before queueing
# ---------------------------------------------------------------------------


def test_submit_allocates_chain_sized_by_payload_byte_length():
    """n_blocks = max(1, ceil(byte_len/64)); byte_len = utf-8 encoded length."""
    sup = make_supervisor("w1")
    bt = sup.block_table
    free0 = bt.num_free
    sup.submit("", worker_id="w1")  # 0 bytes -> minimum 1 block
    r1 = sup.last_request_id
    assert bt.chain_len(sup.kv_cache[r1]) == 1
    sup.submit("a" * 64, worker_id="w1")  # exactly one full block -> 1 block
    r2 = sup.last_request_id
    assert bt.chain_len(sup.kv_cache[r2]) == 1
    sup.submit("a" * 65, worker_id="w1")  # 1 byte over -> 2 blocks
    r3 = sup.last_request_id
    assert bt.chain_len(sup.kv_cache[r3]) == 2
    sup.submit("é" * 50, worker_id="w1")  # UTF-8: 50 chars = 100 BYTES -> 2 blocks
    r4 = sup.last_request_id
    assert bt.chain_len(sup.kv_cache[r4]) == 2
    assert sup.chain_capacity[r4] == 128  # capacity bookkeeping in tokens
    assert bt.num_free == free0 - (1 + 1 + 2 + 2)
    assert sup.last_request_id == 4  # monotonic request counter


def test_crash_mid_submit_frees_chain_before_safe_queueing():
    """WorkerCrash mid-flight leaks ZERO blocks: num_free identical after."""
    sup = Supervisor()
    sup.add_worker("w1", fail_next=True)
    bt = sup.block_table
    free0 = bt.num_free
    res = sup.submit("z" * 300)  # mid-flight chain would be ceil(300/64)=5 blocks
    assert res.startswith(SAFE_QUEUE_PREFIX)  # documented sentinel choice
    assert bt.num_free == free0  # every block of the crashed chain is back
    assert sup.kv_cache == {}  # no dangling supervisor-owned ownership entry
    assert sup.safe_queue == ["z" * 300]  # queued AFTER the chain was freed
    assert sup.last_request_id == 1  # monotonic counter advanced even on crash
    head = bt.allocate_chain(5)  # freed blocks are genuinely reusable
    assert bt.chain_len(head) == 5


def test_expand_request_grows_chain_len_by_ceil_of_extra_bytes():
    """[EXPAND] supervisor-side: ceil(extra_bytes/64) new blocks per call."""
    sup = make_supervisor("w1")
    sup.submit("1234567890", worker_id="w1")  # 10 bytes -> 1 block
    rid = sup.last_request_id
    head = sup.kv_cache[rid]
    bt = sup.block_table
    assert bt.chain_len(head) == 1
    assert sup.chain_capacity[rid] == 64
    assert sup.expand_request(rid, 1) is not None  # ceil(1/64) = 1 new block
    assert bt.chain_len(head) == 2
    sup.expand_request(rid, 129)  # ceil(129/64) = 3 new blocks
    assert bt.chain_len(head) == 5
    assert sup.chain_capacity[rid] == 5 * 64
    assert len(bt.page_table_row(head)) == 5  # chain still fully linked


def test_release_frees_whole_chain_and_double_release_raises():
    """release(): every block back to free list; double release raises."""
    sup = make_supervisor("w1")
    bt = sup.block_table
    free0 = bt.num_free
    sup.submit("x" * 200, worker_id="w1")  # 4 blocks
    rid = sup.last_request_id
    sup.expand_request(rid, 100)  # +2 blocks -> chain of 6
    head = sup.kv_cache[rid]
    assert bt.chain_len(head) == 6
    assert bt.num_free == free0 - 6
    sup.release(rid)
    assert bt.num_free == free0  # whole chain freed
    assert rid not in sup.kv_cache
    assert rid not in sup.chain_capacity
    with pytest.raises(RuntimeError):
        sup.release(rid)  # double release
    assert bt.allocate_chain(6) >= 0  # blocks genuinely reusable again


def test_expand_and_release_unknown_request_ids_raise():
    sup = make_supervisor("w1")
    with pytest.raises(RuntimeError):
        sup.expand_request(42, 64)
    with pytest.raises(RuntimeError):
        sup.release(42)
    sup.submit("abc", worker_id="w1")
    rid = sup.last_request_id
    with pytest.raises(RuntimeError):
        sup.expand_request(rid, 0)  # nonsensical expansion
    with pytest.raises(RuntimeError):
        sup.expand_request(rid, -5)
