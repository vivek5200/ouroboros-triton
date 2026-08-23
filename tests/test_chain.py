"""Tests for table-level SEQUENCE operations on BlockTable.

These cover the [EXPAND]-aware chain primitives consumed by the future
kernel dispatch loop: allocate_chain / walk / chain_len / expand_chain.
Pure bookkeeping — no torch, no triton.
"""

import pytest

from src.block_table import BlockTable


# ---------------------------------------------------------------------------
# allocate_chain
# ---------------------------------------------------------------------------


def test_allocate_chain_walk_order_is_head_to_tail_ascending():
    bt = BlockTable(max_blocks=16)
    head = bt.allocate_chain(3)
    walked = list(bt.walk(head))
    assert len(walked) == 3
    # Ascending allocation ids: fresh slots hand out 0,1,2 so a 3-chain
    # allocated from an empty table walks [head .. tail] in ascending order.
    assert walked == sorted(walked)
    assert walked == [head, head + 1, head + 2]
    assert bt.next_ptr[walked[-1]] == -1          # tail marker intact
    assert all(bt.next_ptr[w] != -1 for w in walked[:-1])  # interior links set


def test_allocate_chain_returns_head_and_consumes_exactly_n():
    bt = BlockTable(max_blocks=8)
    free_before = bt.num_free
    head = bt.allocate_chain(5)
    assert bt.num_free == free_before - 5
    assert list(bt.walk(head)) == list(range(head, head + 5))


def test_allocate_chain_single_block_is_valid_chain():
    bt = BlockTable(max_blocks=4)
    head = bt.allocate_chain(1)
    assert list(bt.walk(head)) == [head]
    assert bt.chain_len(head) == 1


# ---------------------------------------------------------------------------
# walk / chain_len
# ---------------------------------------------------------------------------


def test_walk_yields_ids_via_next_ptr_until_minus_one():
    bt = BlockTable(max_blocks=8)
    ids = [bt.allocate_block() for _ in range(4)]
    for prev, nxt in zip(ids, ids[1:]):
        bt.link(prev, nxt)
    assert list(bt.walk(ids[0])) == ids


def test_chain_len_counts_every_block_in_chain():
    bt = BlockTable(max_blocks=32)
    head = bt.allocate_chain(7)
    assert bt.chain_len(head) == 7
    tail = bt.expand_chain(head + 6)
    assert bt.chain_len(head) == 8                # grew by one
    assert tail == head + 7


# ---------------------------------------------------------------------------
# expand_chain ([EXPAND] primitive)
# ---------------------------------------------------------------------------


def test_expand_chain_grows_walk_by_exactly_one_and_returns_new_tail():
    bt = BlockTable(max_blocks=16)
    head = bt.allocate_chain(3)
    old = list(bt.walk(head))
    new_tail = bt.expand_chain(old[-1])
    grown = list(bt.walk(head))
    assert len(grown) == len(old) + 1             # exactly one longer
    assert grown[:-1] == old                      # prefix unchanged
    assert grown[-1] == new_tail                  # new id appended at end
    assert bt.next_ptr[old[-1]] == new_tail       # old tail now points at it
    assert bt.next_ptr[new_tail] == -1            # new block is the tail


def test_expand_chain_repeated_calls_build_incrementally():
    """Dispatch-loop pattern: grow one block at a time from current tail."""
    bt = BlockTable(max_blocks=64)
    tail = bt.allocate_chain(1)
    head = tail
    for expected_len in range(2, 11):
        tail = bt.expand_chain(tail)
        assert bt.chain_len(head) == expected_len
        assert tail == list(bt.walk(head))[-1]


# ---------------------------------------------------------------------------
# Failure modes: state must be UNCHANGED after every failed call
# ---------------------------------------------------------------------------


def test_allocate_chain_zero_length_raises():
    bt = BlockTable(max_blocks=4)
    with pytest.raises(RuntimeError):
        bt.allocate_chain(0)
    assert bt.num_free == 4                       # nothing consumed
    # Table still fully usable afterwards.
    assert bt.chain_len(bt.allocate_chain(1)) == 1


def test_allocate_chain_negative_length_raises():
    bt = BlockTable(max_blocks=4)
    with pytest.raises(RuntimeError):
        bt.allocate_chain(-3)
    assert bt.num_free == 4


def test_allocate_chain_exhaustion_leaves_state_unchanged():
    bt = BlockTable(max_blocks=4)
    head = bt.allocate_chain(2)
    free_before = bt.num_free
    next_before = list(bt.next_ptr)
    blocks_before = [list(b) if b is not None else None for b in bt.blocks]
    with pytest.raises(RuntimeError):
        bt.allocate_chain(3)                      # only 2 free < 3 needed
    assert bt.num_free == free_before             # pool untouched
    assert bt.next_ptr == next_before             # no partial linking
    assert [b if b is None else list(b) for b in bt.blocks] == blocks_before
    assert bt.chain_len(head) == 2                # existing chain intact


def test_expand_chain_pool_exhaustion_leaves_state_unchanged():
    bt = BlockTable(max_blocks=3)
    head = bt.allocate_chain(3)                   # drain the pool
    tail = head + 2
    snapshot = (
        bt.num_free,
        list(bt.next_ptr),
        [None if b is None else list(b) for b in bt.blocks],
    )
    with pytest.raises(RuntimeError):
        bt.expand_chain(tail)
    assert (bt.num_free, bt.next_ptr, bt.blocks) == (
        snapshot[0],
        snapshot[1],
        snapshot[2],
    )
    assert bt.chain_len(head) == 3                # chain not corrupted


def test_expand_chain_on_freed_tail_raises():
    bt = BlockTable(max_blocks=8)
    head = bt.allocate_chain(2)
    tail = head + 1
    bt.free_block(tail)
    free_before = bt.num_free
    with pytest.raises(RuntimeError):
        bt.expand_chain(tail)
    assert bt.num_free == free_before             # failed call consumed nothing


# ---------------------------------------------------------------------------
# Interaction with freed blocks / slot identity invariant
# ---------------------------------------------------------------------------


def test_walk_from_freed_head_raises():
    bt = BlockTable(max_blocks=8)
    head = bt.allocate_chain(3)
    bt.free_block(head)
    with pytest.raises(RuntimeError):
        list(bt.walk(head))


def test_walk_stops_cleanly_at_freed_midchain_block():
    """Freeing an interior block makes its successor unreachable via walk —
    the generator itself must never crash mid-iteration on valid ids."""
    bt = BlockTable(max_blocks=8)
    head = bt.allocate_chain(3)
    mid, last = head + 1, head + 2
    # Rewire around the freed block first (caller's job), then walk.
    bt.link(head, last)
    bt.free_block(mid)
    assert list(bt.walk(head)) == [head, last]


def test_chains_survive_alloc_free_cycles_with_identity_intact():
    bt = BlockTable(max_blocks=4)
    h1 = bt.allocate_chain(2)                     # ids 0,1
    h2 = bt.allocate_chain(2)                     # ids 2,3
    assert bt.chain_len(h1) == 2 and bt.chain_len(h2) == 2
    bt.free_block(h1)
    bt.free_block(h1 + 1)
    assert bt.num_free == 2
    h3 = bt.allocate_chain(2)                     # reuses exactly ids {0,1}
    # Free list is LIFO (stack), so ids come back in reverse free order;
    # slot IDENTITY is what must hold: id == storage index forever.
    assert sorted(bt.walk(h3)) == [h1, h1 + 1]    # same slots, fresh payload
    assert bt.chain_len(h3) == 2
    assert bt.get_block(h3) == [0] * 64           # zeroed on realloc
