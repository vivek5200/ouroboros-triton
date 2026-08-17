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
