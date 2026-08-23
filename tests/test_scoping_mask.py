"""Tests for the triton-side Table-1 scoping mask (Module 4 port).

:mod:`src.scoping_mask` mirrors ``ouroboros-core/src/scoping.py``
case-for-case so both repos agree on Paper Table 1 semantics. These
tests replicate core's key cases as LITERALS (hand-written matrices,
not derived from the implementation):

* exact 8x8 mask for 1 GLOBAL + 2 LOCAL-sibling spans;
* the deliberate asymmetry G-row/L-col=0 vs L-row/G-col=1;
* partition ValueErrors (gap/overlap/non-zero-start/zero-length/
  negative/unknown-scope);
* the L = 0 edge;
* the assert-style ``global_rows_frozen`` predicate.

No torch / numpy anywhere, matching the golden-reference test style.
"""

import copy

import pytest

from src.scoping_mask import (
    GLOBAL,
    LOCAL,
    Scope,
    block_sparse_mask,
    build_scope_tags,
    global_rows_frozen,
    same_local_scope,
)

G = "global"
L = "local"

# Tiny canonical fixture: 1 GLOBAL span + 2 LOCAL sibling spans (8 tokens).
#   tokens:  0 1 | 2 3 4 | 5 6 7
#   scopes:  G G | L L L | L L L      (A = [2,5), B = [5,8))
SPANS_8 = [(0, 2, G), (2, 5, L), (5, 8, L)]

# Hand-written literal expected mask for SPANS_8 (Table 1 applied by hand).
EXPECTED_8 = [
    # j:    0      1      2      3      4      5      6      7
    [True,  True,  False, False, False, False, False, False],  # i=0 (G)
    [True,  True,  False, False, False, False, False, False],  # i=1 (G)
    [True,  True,  True,  True,  True,  False, False, False],  # i=2 (A)
    [True,  True,  True,  True,  True,  False, False, False],  # i=3 (A)
    [True,  True,  True,  True,  True,  False, False, False],  # i=4 (A)
    [True,  True,  False, False, False, True,  True,  True ],  # i=5 (B)
    [True,  True,  False, False, False, True,  True,  True ],  # i=6 (B)
    [True,  True,  False, False, False, True,  True,  True ],  # i=7 (B)
]


# ---------------------------------------------------------------------------
# Scope constants
# ---------------------------------------------------------------------------

def test_scope_constants_match_core():
    assert Scope.GLOBAL == GLOBAL == "global"
    assert Scope.LOCAL == LOCAL == "local"


# ---------------------------------------------------------------------------
# build_scope_tags
# ---------------------------------------------------------------------------

def test_build_scope_tags_basic():
    assert build_scope_tags(SPANS_8) == [
        G, G, L, L, L, L, L, L,
    ]


def test_build_scope_tags_accepts_unordered_exact_partition():
    assert build_scope_tags([(2, 5, L), (5, 8, L), (0, 2, G)]) == [
        G, G, L, L, L, L, L, L,
    ]


@pytest.mark.parametrize(
    "bad",
    [
        [(0, 2, G), (3, 5, L)],           # gap between regions
        [(0, 3, G), (2, 5, L)],           # overlap between regions
        [(1, 4, L)],                      # does not start at 0
        [(0, 0, G)],                      # zero-length region
        [(-1, 2, G)],                     # negative start
        [(0, 4, "friend")],               # unknown scope name
    ],
)
def test_build_scope_tags_partition_violation_raises_valueerror(bad):
    with pytest.raises(ValueError):
        build_scope_tags(bad)


def test_build_scope_tags_empty_l0_edge():
    assert build_scope_tags([]) == []


# ---------------------------------------------------------------------------
# same_local_scope
# ---------------------------------------------------------------------------

def test_same_local_scope_pairs():
    assert same_local_scope(2, 4, SPANS_8) is True       # within local A
    assert same_local_scope(3, 3, SPANS_8) is True       # self inside local
    assert same_local_scope(2, 6, SPANS_8) is False      # siblings A/B
    assert same_local_scope(4, 5, SPANS_8) is False      # siblings A/B
    assert same_local_scope(0, 1, SPANS_8) is False      # globals never local
    assert same_local_scope(0, 0, SPANS_8) is False
    assert same_local_scope(0, 2, SPANS_8) is False      # mixed G/L


def test_same_local_scope_out_of_range_is_false():
    assert same_local_scope(-1, 2, SPANS_8) is False
    assert same_local_scope(2, 8, SPANS_8) is False
    assert same_local_scope(0, 0, []) is False


# ---------------------------------------------------------------------------
# block_sparse_mask — exact 8x8 literal + edges
# ---------------------------------------------------------------------------

def test_block_sparse_mask_8token_exact_literal():
    assert block_sparse_mask(SPANS_8) == EXPECTED_8


def test_block_sparse_mask_entries_are_real_bools():
    mask = block_sparse_mask(SPANS_8)
    assert len(mask) == 8 and all(len(row) == 8 for row in mask)
    assert all(isinstance(v, bool) for row in mask for v in row)


def test_block_sparse_mask_l0_edge():
    assert block_sparse_mask([]) == []


def test_asymmetry_global_row_local_col_zero_vs_local_row_global_col_one():
    """LOCAL row x GLOBAL col = 1 while mirrored GLOBAL row x LOCAL col = 0."""
    mask = block_sparse_mask(SPANS_8)
    assert mask[2][0] is True            # body sees signature
    assert mask[0][2] is False           # global KV insulated from locals
    assert mask[2][0] != mask[0][2]
    assert mask[7][1] is True and mask[1][7] is False


def test_table1_rules_hold_for_every_cell_spans8():
    """Exhaustive per-cell check of Table 1 (+ documented GLOBALxGLOBAL=1)."""
    tags = build_scope_tags(SPANS_8)
    owners = []
    for s, e, sc in sorted(SPANS_8):
        owners.extend([(s, e)] * (e - s))
    mask = block_sparse_mask(SPANS_8)
    for i in range(8):
        for j in range(8):
            if i == j:
                expected = True                       # self
            elif tags[i] == L and tags[j] == G:
                expected = True                       # body sees sig
            elif tags[i] == L and tags[j] == L:
                expected = owners[i] == owners[j]     # same scope only
            else:  # i is GLOBAL
                expected = tags[j] == G               # insulated from L
            assert mask[i][j] == expected, (i, j)


# ---------------------------------------------------------------------------
# global_rows_frozen — assert-style cache-coherency check (never raises)
# ---------------------------------------------------------------------------

def _expanded_world():
    """2 GLOBAL tokens then one LOCAL span grown [2,4)->[2,6) by splicing."""
    spans_before = [(0, 2, G), (2, 4, L)]
    spans_after = [(0, 2, G), (2, 6, L)]
    before = block_sparse_mask(spans_before)
    after = block_sparse_mask(spans_after)
    return before, after, spans_after


def test_global_rows_frozen_true_for_clean_expand():
    before, after, spans_after = _expanded_world()
    assert global_rows_frozen(before, after, spans_after) is True


def test_global_rows_frozen_false_on_corrupted_global_row():
    before, after, spans_after = _expanded_world()
    bad = copy.deepcopy(after)
    bad[0][4] = True   # a global row gained attention toward a LOCAL token
    assert global_rows_frozen(before, bad, spans_after) is False


def test_global_rows_frozen_false_on_shape_mismatch_never_raises():
    before, after, spans_after = _expanded_world()
    assert global_rows_frozen(before, after[1:], spans_after) is False
    assert global_rows_frozen(before, [], spans_after) is False


def test_global_rows_frozen_swallows_malformed_spans():
    before, after, _ = _expanded_world()
    with_gap = [(0, 2, G), (3, 6, L)]     # does not partition -> not frozen
    assert global_rows_frozen(before, after, with_gap) is False
