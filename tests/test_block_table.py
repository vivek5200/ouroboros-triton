"""Tests for the Block Table module."""

import pytest
from src.block_table import BlockTable, BLOCK_SIZE


def test_block_size_is_64():
    assert BLOCK_SIZE == 64


def test_allocate_block():
    bt = BlockTable(max_blocks=10)
    idx = bt.allocate_block()
    assert idx == 0
    assert bt.num_free == 9


def test_free_block():
    bt = BlockTable(max_blocks=10)
    idx = bt.allocate_block()
    bt.free_block(idx)
    assert bt.num_free == 10


def test_allocate_exhaustion():
    bt = BlockTable(max_blocks=1)
    bt.allocate_block()
    with pytest.raises(RuntimeError):
        bt.allocate_block()


# ---------------------------------------------------------------------------
# Regression tests: slot-identity invariant (post Phase 0.5 survey fix)
# ---------------------------------------------------------------------------


def test_realloc_reuses_freed_slot_identity():
    """alloc → free → alloc must reuse the freed id with FRESH storage."""
    bt = BlockTable(max_blocks=4)
    a = bt.allocate_block()              # 0
    bt.get_block(a)[5] = 42              # payload in slot a
    b = bt.allocate_block()              # 1
    bt.get_block(b)[5] = 99

    bt.free_block(a)
    c = bt.allocate_block()
    assert c == a                        # freed id reused...
    assert bt.get_block(c) == [0] * BLOCK_SIZE  # ...zeroed, not stale
    assert bt.get_block(b)[5] == 99      # neighbor untouched


def test_data_isolation_between_blocks():
    bt = BlockTable(max_blocks=4)
    x = bt.allocate_block()
    y = bt.allocate_block()
    assert x != y
    bt.get_block(x)[0] = 111
    assert bt.get_block(y)[0] == 0       # no aliasing between slots


def test_double_free_raises():
    bt = BlockTable(max_blocks=2)
    i = bt.allocate_block()
    bt.free_block(i)
    with pytest.raises(RuntimeError):
        bt.free_block(i)


def test_get_block_unallocated_raises():
    bt = BlockTable(max_blocks=2)
    with pytest.raises(RuntimeError):
        bt.get_block(1)                  # never allocated


def test_link_sets_next_ptr():
    bt = BlockTable(max_blocks=4)
    a = bt.allocate_block()
    b = bt.allocate_block()
    bt.link(a, b)
    assert bt.next_ptr[a] == b           # array-backed linked list pointer
    assert bt.next_ptr[b] == -1          # tail marker
