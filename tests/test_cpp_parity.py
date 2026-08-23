"""Differential parity: production C++ BlockTable vs Python reference (law sys-blocks).

Strategy — same seeds MUST give identical observable behaviour:

* A seeded RNG generates an explicit **op script** up front (pure Python RNG,
  so both implementations see byte-identical inputs regardless of language).
* The script is executed in lockstep against ``src.block_table.BlockTable``
  and the pybind11 ``ouroboros_cpp.BlockTable``; at every step the two must
  agree on return value AND on whether a RuntimeError was raised.
* After each script, the full observable state (free count, next_ptr array,
  payloads, walks) is compared.

The extension is built ONCE per test session by running
``src/cpp/build.sh`` via subprocess; if the toolchain is unavailable the
tests SKIP with the builder's own reason instead of failing.
"""

import importlib.util
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_DIR = REPO_ROOT / "src" / "cpp"
BUILD_SH = CPP_DIR / "build.sh"

from src.block_table import BLOCK_SIZE, PAD_ID, BlockTable as PyBlockTable


# ---------------------------------------------------------------------------
# Module fixture: build-once attempt, then import; skip when unavailable
# ---------------------------------------------------------------------------


def _find_extension():
    return sorted(CPP_DIR.glob("ouroboros_cpp*.so"))


def _import_extension(path: Path):
    spec = importlib.util.spec_from_file_location("ouroboros_cpp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def ouroboros_cpp():
    existing = _find_extension()
    build_log = "(extension already built)"
    if not existing:
        if not BUILD_SH.exists():
            pytest.skip(f"{BUILD_SH} missing")
        proc = subprocess.run(
            ["bash", str(BUILD_SH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        build_log = (proc.stdout + proc.stderr).strip()
        existing = _find_extension()
    if not existing:
        pytest.skip(
            "ouroboros_cpp extension not built — install pybind11 + g++ "
            f"and run bash src/cpp/build.sh. Builder said: {build_log}"
        )
    try:
        return _import_extension(existing[0])
    except ImportError as exc:
        pytest.skip(f"ouroboros_cpp{existing[0].suffix} failed to import: {exc}")


# ---------------------------------------------------------------------------
# Lockstep execution engine
# ---------------------------------------------------------------------------

_MISSING = object()


def _call(fn):
    """Run fn; return (result, error_type_name) where errors collapse to a tag."""
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - parity must hold for ANY failure
        return _MISSING, type(exc).__name__


def _assert_agree(label, py_fn, cpp_fn):
    py_res, py_err = _call(py_fn)
    cpp_res, cpp_err = _call(cpp_fn)
    assert py_err == cpp_err, (
        f"{label}: Python raised {py_err}, C++ raised {cpp_err}"
    )
    assert py_res == cpp_res, f"{label}: Python {py_res!r} != C++ {cpp_res!r}"
    return py_res


class Lockstep:
    """Drives one op script through both implementations simultaneously."""

    def __init__(self, max_blocks: int = 24):
        self.max_blocks = max_blocks
        self.py = PyBlockTable(max_blocks=max_blocks)
        self.cpp_tbl = None  # set by run()
        self.allocated = []  # ids currently allocated (identical both sides)

    def run(self, cpp_cls):
        self.cpp_tbl = cpp_cls(self.max_blocks)
        return self

    def _pick_allocated(self, rng):
        return rng.choice(self.allocated) if self.allocated else 0

    def _note_alloc(self, idx):
        self.allocated.append(idx)

    def _note_free(self, idx):
        self.allocated.remove(idx)

    # -- individual ops: each compares result + exception parity ------------

    def op_allocate(self, rng):
        idx = _assert_agree(
            "allocate_block",
            lambda: self.py.allocate_block(),
            lambda: self.cpp_tbl.allocate_block(),
        )
        self._note_alloc(idx)

    def op_write(self, rng):
        idx = self._pick_allocated(rng)
        off, val = rng.randrange(BLOCK_SIZE), rng.randrange(-1000, 1000)
        _assert_agree(
            f"set_cell({idx},{off})",
            lambda: self.py.get_block(idx).__setitem__(off, val),
            lambda: self.cpp_tbl.set_cell(idx, off, val),
        )

    def op_read(self, rng):
        idx = self._pick_allocated(rng)
        off = rng.randrange(BLOCK_SIZE)
        got = _assert_agree(
            f"get_cell({idx},{off})",
            lambda: self.py.get_block(idx)[off],
            lambda: self.cpp_tbl.get_cell(idx, off),
        )
        return got

    def op_free(self, rng):
        idx = self._pick_allocated(rng)
        _assert_agree(
            f"free_block({idx})",
            lambda: self.py.free_block(idx),
            lambda: self.cpp_tbl.free_block(idx),
        )
        self._note_free(idx)

    def op_link(self, rng):
        a, b = self._pick_allocated(rng), self._pick_allocated(rng)
        _assert_agree(
            f"link({a},{b})",
            lambda: self.py.link(a, b),
            lambda: self.cpp_tbl.link(a, b),
        )

    def op_chain(self, rng):
        k = rng.randint(1, 4)
        head = _assert_agree(
            f"allocate_chain({k})",
            lambda: self.py.allocate_chain(k),
            lambda: self.cpp_tbl.allocate_chain(k),
        )
        if isinstance(head, int):
            # Chain landed identically on both sides; track its fresh ids.
            for wid in list(self.py.walk(head)):
                if wid not in self.allocated:
                    self.allocated.append(wid)

    def op_expand(self, rng):
        tail = self._pick_allocated(rng)
        new_tail = _assert_agree(
            f"expand_chain({tail})",
            lambda: self.py.expand_chain(tail),
            lambda: self.cpp_tbl.expand_chain(tail),
        )
        if isinstance(new_tail, int) and new_tail not in self.allocated:
            self.allocated.append(new_tail)

    def op_walk(self, rng):
        head = self._pick_allocated(rng)
        _assert_agree(
            f"walk({head})",
            lambda: list(self.py.walk(head)),
            lambda: self.cpp_tbl.walk(head),
        )

    def op_chain_len(self, rng):
        head = self._pick_allocated(rng)
        _assert_agree(
            f"chain_len({head})",
            lambda: self.py.chain_len(head),
            lambda: self.cpp_tbl.chain_len(head),
        )

    def op_resolve(self, rng):
        head = self._pick_allocated(rng)
        logical = rng.randint(-1, 6)
        _assert_agree(
            f"resolve_block({head},{logical})",
            lambda: self.py.resolve_block(head, logical),
            lambda: self.cpp_tbl.resolve_block(head, logical),
        )

    def op_page_row(self, rng):
        head = self._pick_allocated(rng)
        pad = rng.choice([None, 3, 6])
        if pad is None:
            _assert_agree(
                f"page_table_row({head})",
                lambda: self.py.page_table_row(head),
                lambda: self.cpp_tbl.page_table_row(head),
            )
        else:
            _assert_agree(
                f"page_table_row({head},pad_to={pad})",
                lambda: self.py.page_table_row(head, pad_to=pad),
                lambda: self.cpp_tbl.page_table_row(head, pad_to=pad),
            )

    def op_invalid_probe(self, rng):
        """Deliberately illegal ops must fail IDENTICALLY on both sides."""
        kind = rng.choice(["double_free", "bad_id", "unallocated_get"])
        if kind == "double_free":
            victim = rng.randrange(self.max_blocks)
            if self.py.blocks[victim] is None:
                _assert_agree(
                    f"double_free({victim})",
                    lambda: self.py.free_block(victim),
                    lambda: self.cpp_tbl.free_block(victim),
                )
        elif kind == "bad_id":
            bad = rng.choice([-1, self.max_blocks, self.max_blocks * 7])
            _assert_agree(
                f"get_block({bad})",
                lambda: self.py.get_block(bad),
                lambda: self.cpp_tbl.get_block(bad),
            )
        else:
            unallocated = next(
                (i for i in range(self.max_blocks) if self.py.blocks[i] is None), 0
            )
            _assert_agree(
                f"link_to_unallocated({unallocated})",
                lambda: self.py.link(0, unallocated),
                lambda: self.cpp_tbl.link(0, unallocated),
            )

    OPS = [
        op_allocate,
        op_allocate,
        op_allocate,
        op_write,
        op_read,
        op_free,
        op_link,
        op_chain,
        op_expand,
        op_walk,
        op_chain_len,
        op_resolve,
        op_page_row,
        op_invalid_probe,
    ]

    def snapshot_mismatch(self):
        """Full observable-state comparison after a script finishes."""
        problems = []
        if self.py.num_free != self.cpp_tbl.num_free:
            problems.append(
                f"num_free: py={self.py.num_free} cpp={self.cpp_tbl.num_free}"
            )
        if list(self.py.next_ptr) != list(self.cpp_tbl.next_ptr):
            problems.append(
                f"next_ptr:\n  py ={list(self.py.next_ptr)}\n  cpp={list(self.cpp_tbl.next_ptr)}"
            )
        py_payload = [b if b is None else list(b) for b in self.py.blocks]
        cpp_payload = [None if b is None else list(b) for b in self.cpp_tbl.blocks]
        if py_payload != cpp_payload:
            diff = [
                i for i, (a, b) in enumerate(zip(py_payload, cpp_payload)) if a != b
            ]
            problems.append(f"payload differs at slots {diff[:8]}")
        for i in range(self.max_blocks):
            if self.py.blocks[i] is not None:
                w_py = list(self.py.walk(i))
                w_cpp = list(self.cpp_tbl.walk(i))
                if w_py != w_cpp:
                    problems.append(f"walk({i}): py={w_py} cpp={w_cpp}")
        return problems


def _run_script(cpp_cls, seed: int, n_ops: int):
    rng = random.Random(seed)
    ls = Lockstep(max_blocks=24).run(cpp_cls)
    for i in range(n_ops):
        op = rng.choice(ls.OPS)
        op(ls, rng)
    problems = ls.snapshot_mismatch()
    assert not problems, f"seed {seed}: state diverged:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# The differential tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_differential_random_op_sequences(ouroboros_cpp, seed):
    _run_script(ouroboros_cpp.BlockTable, seed=seed, n_ops=160)


def test_differential_long_script_single_seed(ouroboros_cpp):
    _run_script(ouroboros_cpp.BlockTable, seed=42, n_ops=600)


def test_cpp_double_free_and_slot_identity(ouroboros_cpp):
    """Direct checks of the invariants the Python suite pins down."""
    bt = ouroboros_cpp.BlockTable(8)
    assert bt.block_size == BLOCK_SIZE
    a = bt.allocate_block()
    assert a == 0 and bt.num_free == 7          # lowest id first, like Python
    bt.set_cell(a, 5, 42)
    assert bt.get_cell(a, 5) == 42
    bt.free_block(a)                             # first free is legal...
    with pytest.raises(RuntimeError):            # ...double free is not
        bt.free_block(a)
    b = bt.allocate_block()
    assert b == a                                # slot identity: id == index
    assert bt.get_block(b) == [0] * BLOCK_SIZE   # fresh zeroed payload
    assert bt.num_free == 7


def test_cpp_pad_id_matches_python(ouroboros_cpp):
    assert ouroboros_cpp.PAD_ID == PAD_ID
