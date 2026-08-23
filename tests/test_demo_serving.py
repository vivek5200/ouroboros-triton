"""TDD contract for scripts/demo_serving.py (Module 6 HA end-to-end demo).

The demo's core is ``run_demo() -> dict``: one realistic supervisor session
(3 workers, one pre-armed to segfault mid-stream) over 12 requests of
varying sizes routed through ``Supervisor.submit()``, followed by
health_pass + restart + drain_safe_queue and full teardown. Every Module-6
HA invariant must show up in the returned dict:

  * served_count      -- all 12 requests end up served exactly once
                         (directly, or queued then drained exactly once)
  * queued_count > 0  -- the forced crash degraded to the Safe Queue
  * leak_free         -- zero blocks leaked anywhere (crash path included):
                         num_free delta == 0 and allocated == freed
  * drain_exact_once  -- FIFO drain re-served each queued request exactly
                         once, queue left empty

Heavy printing lives behind the ``verbose`` flag; tests run it silent.
"""

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "demo_serving.py"

_spec = importlib.util.spec_from_file_location("demo_serving", _SCRIPT)
demo_serving = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_serving)

REQUIRED_KEYS = {
    "total_requests",
    "served_count",
    "queued_count",
    "drained_count",
    "blocks_allocated",
    "blocks_freed",
    "num_free_delta",
    "leak_free",
    "drain_exact_once",
    "kv_consistent",
    "all_pass",
}


def test_run_demo_returns_full_invariant_dict():
    result = demo_serving.run_demo(verbose=False)
    assert isinstance(result, dict)
    missing = REQUIRED_KEYS - set(result)
    assert not missing, f"run_demo() dict missing invariant keys: {missing}"


def test_all_twelve_requests_served_exactly_once():
    result = demo_serving.run_demo(verbose=False)
    assert result["total_requests"] == 12
    assert result["served_count"] == 12


def test_forced_crash_degrades_to_safe_queue_then_drains():
    """The pre-armed worker must crash mid-stream: queued > 0, fully drained."""
    result = demo_serving.run_demo(verbose=False)
    assert result["queued_count"] > 0, "forced crash produced no Safe Queue entry"
    assert result["safe_queue_empty"] is True
    assert result["drained_count"] == result["queued_count"]
    assert result["drain_exact_once"] is True


def test_zero_block_leaks_across_crash_restart_drain_release():
    result = demo_serving.run_demo(verbose=False)
    assert result["crash_zero_leak"] is True, "crash path leaked its in-flight chain"
    assert result["num_free_delta"] == 0
    assert result["blocks_allocated"] == result["blocks_freed"] > 0
    assert result["leak_free"] is True


def test_kv_cache_consistent_and_everything_green():
    result = demo_serving.run_demo(verbose=False)
    assert result["kv_consistent"] is True
    assert result["served_exactly_once"] is True
    assert result["restarted_workers"], "health_pass+restart never ran"
    # The CLI contract: exit 0 iff every invariant holds.
    assert result["all_pass"] is True
