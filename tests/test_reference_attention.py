"""Tests for the pure-python golden reference attention (paper §4.3).

The golden reference in :mod:`src.kernels.reference_attention` is the
stdlib-only oracle the Triton kernel must match bit-for-bounded. These
tests pin its semantics independently of the kernel:

* hand-computed 3-token example (d=2, known rotations) — exact floats;
* relative-offset law (Module 3): scores depend only on q/k position gap;
* scope-mask False column removes ALL weight from that key;
* additive bias shifts attention monotonically (two-key case);
* ``reference_from_block_table`` round-trips a live BlockTable chain.

No torch / numpy / triton anywhere: this file runs on any host.
"""

import math

import pytest

from src.block_table import BlockTable
from src.kernels.reference_attention import (
    golden_attention,
    reference_from_block_table,
    rope_rotate,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _rot(v, angle):
    """Explicit 2-d rotation spelled out here — NOT rope_rotate."""
    c, s = math.cos(angle), math.sin(angle)
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]


def _softmax(xs):
    m = max(xs)
    ws = [math.exp(x - m) for x in xs]
    z = sum(ws)
    return [w / z for w in ws]


# ---------------------------------------------------------------------------
# rope_rotate
# ---------------------------------------------------------------------------


class TestRopeRotate:
    def test_identity_at_position_zero(self):
        assert rope_rotate([1.0, 0.0], 0) == [1.0, 0.0]
        assert rope_rotate([0.5, -0.25], 0) == [0.5, -0.25]

    def test_d2_rotation_angle_equals_position(self):
        # d=2 -> theta_0 = base**0 = 1 rad per position step.
        got = rope_rotate([1.0, 0.0], 2)
        want = _rot([1.0, 0.0], 2.0)
        assert got == pytest.approx(want, rel=1e-15)
        got1 = rope_rotate([0.0, 1.0], 1)
        assert got1 == pytest.approx(_rot([0.0, 1.0], 1.0), rel=1e-15)

    def test_preserves_norm_and_input(self):
        v = [3.0, 4.0]
        out = rope_rotate(v, 7)
        assert math.hypot(*out) == pytest.approx(5.0, rel=1e-12)
        assert v == [3.0, 4.0]  # input untouched

    def test_multi_pair_theta_schedule(self):
        # d=4: pair 0 angle pos*base**0, pair 1 angle pos*base**(-2/4).
        v = [1.0, 0.0, 1.0, 0.0]
        got = rope_rotate(v, 3, base=100.0)
        assert got[0] == pytest.approx(math.cos(3.0), rel=1e-12)
        assert got[1] == pytest.approx(math.sin(3.0), rel=1e-12)
        th1 = 100.0 ** (-2.0 / 4.0)
        assert got[2] == pytest.approx(math.cos(3.0 * th1), rel=1e-12)
        assert got[3] == pytest.approx(math.sin(3.0 * th1), rel=1e-12)

    def test_rejects_odd_length_and_bad_args(self):
        with pytest.raises(RuntimeError, match="even"):
            rope_rotate([1.0, 2.0, 3.0], 0)
        with pytest.raises(RuntimeError, match="position"):
            rope_rotate([1.0, 0.0], True)  # bool is not a position
        with pytest.raises(RuntimeError, match="base"):
            rope_rotate([1.0, 0.0], 1, base=0.0)


# ---------------------------------------------------------------------------
# golden_attention — hand-computed example (d=2, exact arithmetic spelled out)
# ---------------------------------------------------------------------------


class TestHandComputedThreeToken:
    def _world(self):
        """Chain order [7, 3, 5] (scattered ids on purpose), d_head=2.

        Query is logical token i=2 (block 5). With d=2 every theta_0 is
        10000**0 == 1 rad, so token at logical position p is rotated by
        exactly p radians:
            rot(q=[1,0] @ pos 2) = (cos2, sin2)
            rot(k=[1,0] @ pos 0) = (1, 0)      rot(k=[0,1] @ pos 1)=(-sin1, cos1)
            rot(k=[1,0] @ pos 2) = (cos2, sin2)
        """
        return (
            [7, 3, 5],
            {
                7: {"q": [0.0, 0.0], "k": [1.0, 0.0], "v": [1.0, 2.0]},
                3: {"q": [0.0, 0.0], "k": [0.0, 1.0], "v": [3.0, 4.0]},
                5: {"q": [1.0, 0.0], "k": [1.0, 0.0], "v": [5.0, 6.0]},
            },
        )

    def test_exact_output_vector(self):
        pt, blocks = self._world()
        r = math.sqrt(2.0)
        # scores = dot(q_rot, k_rot)/sqrt(d), no bias, full mask:
        s0 = math.cos(2.0) / r          # gap 2-0, k=(1,0)
        s1 = (math.sin(2.0 - 1.0)) / r  # dot(rot2 q,( -sin1,cos1)) = sin(1)
        s2 = 1.0 / r                    # identical rotations -> dot = 1
        w = _softmax([s0, s1, s2])
        want_x = w[0] * 1.0 + w[1] * 3.0 + w[2] * 5.0
        want_y = w[0] * 2.0 + w[1] * 4.0 + w[2] * 6.0
        got = golden_attention(pt, blocks, q_idx=2, bias=None, scope_mask=None)
        assert got == pytest.approx([want_x, want_y], rel=1e-12)

    def test_matches_independent_manual_pipeline(self):
        """Same numbers via a fully independent re-derivation in the test."""
        pt, blocks = self._world()
        q = _rot(blocks[pt[2]]["q"], 2.0)
        keys = [_rot(blocks[pt[j]]["k"], float(j)) for j in range(3)]
        scores = [_dot(q, k) / math.sqrt(2.0) for k in keys]
        w = _softmax(scores)
        want = [
            sum(w[j] * blocks[pt[j]]["v"][d] for j in range(3)) for d in range(2)
        ]
        got = golden_attention(pt, blocks, 2, None, None)
        assert got == pytest.approx(want, rel=1e-12)

    def test_query_index_selects_other_row(self):
        pt, blocks = self._world()
        # q_idx=0 -> q vector [0,0] at pos 0 => all scores 0 => uniform weights.
        got = golden_attention(pt, blocks, 0, None, None)
        assert got == pytest.approx([(1 + 3 + 5) / 3.0, (2 + 4 + 6) / 3.0], rel=1e-12)

    def test_rejects_bad_chain_and_unknown_block(self):
        pt, blocks = self._world()
        with pytest.raises(RuntimeError, match="empty"):
            golden_attention([], blocks, 0, None, None)
        with pytest.raises(RuntimeError, match="page table"):
            golden_attention([7, 99], blocks, 0, None, None)
        with pytest.raises(RuntimeError, match="q_idx"):
            golden_attention(pt, blocks, 3, None, None)


# ---------------------------------------------------------------------------
# Relative-offset law (Module 3): score(i, j) depends only on (i - j)
# ---------------------------------------------------------------------------


class TestRelativeOffsetLaw:
    Q = [0.3, -0.7]
    K = [0.6, 0.8]

    def test_same_gap_equal_scores(self):
        from src.kernels.reference_attention import roped_dot

        for i, j in [(2, 0), (3, 1), (5, 3), (9, 7)]:
            assert roped_dot(self.Q, self.K, i, j) == pytest.approx(
                roped_dot(self.Q, self.K, 2, 0), rel=1e-12
            )

    def test_different_gap_different_score(self):
        from src.kernels.reference_attention import roped_dot

        assert roped_dot(self.Q, self.K, 2, 0) != pytest.approx(
            roped_dot(self.Q, self.K, 2, 2), rel=1e-6
        )

    def test_uniform_shift_leaves_output_invariant(self):
        """Translating every logical position by +1 changes nothing."""
        kv = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        vv = [2.0, -1.0, 0.5]
        blocks = {
            10 + n: {"q": [0.4, 0.2], "k": list(kv[n]), "v": [vv[n], vv[n]]}
            for n in range(3)
        }
        out_base = golden_attention(
            [10, 11, 12], dict(blocks), 2, None, None, position_base=77.0
        )
        # Same tokens one position later: a dummy block at slot 0 is masked
        # off, kept keys sit at positions 1..3 instead of 0..2.
        shifted = dict(blocks)
        shifted[99] = {"q": [9.0, 9.0], "k": [9.0, 9.0], "v": [0.0, 0.0]}
        mask = [[True] * 4 for _ in range(4)]
        mask[3][0] = False  # hide the dummy key from query 3
        out_shifted = golden_attention(
            [99, 10, 11, 12],
            shifted,
            3,
            None,
            mask,
            position_base=77.0,
        )
        assert out_shifted == pytest.approx(out_base, rel=1e-10)


# ---------------------------------------------------------------------------
# Scope mask: a False column removes all weight from that key
# ---------------------------------------------------------------------------


class TestScopeMask:
    def _world(self):
        # middle key dominates when unmasked (huge v component)
        pt = [4, 9, 2]
        blocks = {
            4: {"q": [1.0, 0.0], "k": [1.0, 0.0], "v": [1.0, 0.0]},
            9: {"q": [0.0, 0.0], "k": [0.9, 0.1], "v": [50.0, -7.0]},
            2: {"q": [0.0, 0.0], "k": [0.8, -0.2], "v": [-1.0, 1.0]},
        }
        return pt, blocks

    def test_masked_last_column_equals_two_key_chain(self):
        pt, blocks = self._world()
        masked = golden_attention(
            pt, blocks, 0, None, [[True, True, False]]
        )
        reduced = golden_attention([4, 9], {4: blocks[4], 9: blocks[9]}, 0, None, None)
        assert masked == pytest.approx(reduced, rel=1e-12)
        unmasked = golden_attention(pt, blocks, 0, None, None)
        assert masked != pytest.approx(unmasked, rel=1e-3)

    def test_int_mask_semantics_match_kernel(self):
        """Kernel masks are int8 0/nonzero — same contract here."""
        pt, blocks = self._world()
        bool_m = golden_attention(pt, blocks, 0, None, [[True, True, False]])
        int_m = golden_attention(pt, blocks, 0, None, [[1, 1, 0]])
        assert int_m == pytest.approx(bool_m, rel=1e-15)

    def test_fully_masked_query_raises(self):
        pt, blocks = self._world()
        with pytest.raises(RuntimeError, match="fully masked"):
            golden_attention(pt[:2], blocks, 0, None, [[False, False]])


# ---------------------------------------------------------------------------
# Additive bias shifts attention monotonically (two-key case)
# ---------------------------------------------------------------------------


class TestBiasMonotone:
    def _two_key(self):
        pt = [6, 1]
        blocks = {
            6: {"q": [1.0, 0.0], "k": [1.0, 0.0], "v": [1.0, 0.0]},
            1: {"q": [0.0, 0.0], "k": [1.0, 0.0], "v": [0.0, 1.0]},
        }
        return pt, blocks

    def test_bias_on_second_key_pulls_output_monotonically(self):
        pt, blocks = self._two_key()
        ys = []
        for b in (-3.0, -1.0, 0.0, 1.0, 3.0):
            out = golden_attention(pt, blocks, 0, [[0.0, b]], None)
            ys.append(out[1])
        assert all(y2 > y1 for y1, y2 in zip(ys, ys[1:]))

    def test_zero_bias_equals_no_bias(self):
        pt, blocks = self._two_key()
        none = golden_attention(pt, blocks, 0, None, None)
        zero = golden_attention(pt, blocks, 0, [[0.0, 0.0]], None)
        assert zero == pytest.approx(none, rel=1e-15)

    def test_bias_is_additive_not_multiplicative(self):
        pt, blocks = self._two_key()
        # v=(1,0),(0,1) makes weights directly readable from the output:
        # out == [w0, w1]. Additivity (law math-rope) means adding b to key
        # 1's logit shifts log(w1/w0) by EXACTLY b, whatever the base scores.
        def log_odds(bias_b):
            out = golden_attention(pt, blocks, 0, [[0.0, bias_b]], None)
            return math.log(out[1] / out[0])

        assert (log_odds(2.0) - log_odds(0.0)) == pytest.approx(2.0, rel=1e-12)
        assert (log_odds(-1.5) - log_odds(0.0)) == pytest.approx(-1.5, rel=1e-12)


# ---------------------------------------------------------------------------
# Adapter: BlockTable chains -> reference inputs
# ---------------------------------------------------------------------------


class TestReferenceFromBlockTable:
    def test_round_trip_chain_of_three(self):
        bt = BlockTable(max_blocks=32)
        head = bt.allocate_chain(3)
        pt_list, blocks = reference_from_block_table(bt, head, d_head=2)
        assert pt_list == list(bt.walk(head))
        assert len(pt_list) == 3
        assert set(blocks) == set(pt_list)
        for pid in pt_list:
            payload = blocks[pid]
            assert set(payload) == {"q", "k", "v"}
            for vec in payload.values():
                assert len(vec) == 2
                assert all(isinstance(x, float) and math.isfinite(x) for x in vec)

    def test_deterministic_across_calls(self):
        bt = BlockTable(max_blocks=32)
        head = bt.allocate_chain(3)
        a = reference_from_block_table(bt, head, 4)
        b = reference_from_block_table(bt, head, 4)
        assert a == b

    def test_survives_expand_chain_growth(self):
        bt = BlockTable(max_blocks=32)
        head = bt.allocate_chain(3)
        tail = list(bt.walk(head))[-1]
        new_tail = bt.expand_chain(tail)
        pt_list, blocks = reference_from_block_table(bt, head, 2)
        assert pt_list[-1] == new_tail and len(pt_list) == 4
        assert new_tail in blocks

    def test_golden_attention_runs_on_adapter_output(self):
        bt = BlockTable(max_blocks=32)
        h = bt.allocate_chain(3)
        pt_list, blocks = reference_from_block_table(bt, h, 2)
        out = golden_attention(pt_list, blocks, 1, None, None)
        assert len(out) == 2 and all(math.isfinite(x) for x in out)

    def test_rejects_odd_dhead_and_unallocated_head(self):
        bt = BlockTable(max_blocks=32)
        with pytest.raises(RuntimeError, match="even"):
            reference_from_block_table(bt, 0, d_head=3)
        with pytest.raises(RuntimeError, match="allocated|not allocated"):
            reference_from_block_table(bt, 5, d_head=2)


# ---------------------------------------------------------------------------
# End-to-end: golden_attention x Table-1 scoping mask (src.scoping_mask)
# ---------------------------------------------------------------------------

from src.scoping_mask import block_sparse_mask  # noqa: E402  (kept append-only)


class TestGoldenAttentionWithTable1ScopeMask:
    """6-token chain scoped by ``block_sparse_mask`` (core Table-1 port).

    Layout: tokens 0-1 GLOBAL, [2,4) LOCAL span A, [4,6) LOCAL span B.
    V is the identity matrix (v_j = e_j, d_head = 6), which makes every
    output coordinate literally an attention weight: out[d] == w_d. So
    "masked key receives zero weight" is checkable EXACTLY, and the full
    output must equal a manual softmax-reduction over the unmasked subset.
    """

    SPANS_6 = [(0, 2, "global"), (2, 4, "local"), (4, 6, "local")]

    def _world(self):
        pt = [10, 3, 7, 12, 1, 8]  # scattered physical ids, chain order
        qs = [[0.5, -0.2, 0.9, 0.1, -0.4, 0.3],
              [0.8, 0.6, -0.1, 0.2, 0.5, -0.7]]
        ks = [[0.4, 0.9, -0.3, 0.6, 0.2, -0.5],
              [-0.6, 0.1, 0.7, -0.2, 0.8, 0.3],
              [0.9, -0.4, 0.5, 0.1, -0.8, 0.2],
              [0.2, 0.3, -0.9, 0.4, 0.6, 0.7],
              [-0.1, 0.5, 0.2, -0.6, 0.9, -0.3],
              [0.7, -0.8, 0.4, 0.3, -0.2, 0.6]]
        blocks = {}
        for j, pid in enumerate(pt):
            blocks[pid] = {
                # queries alternate between the two GLOBAL-tagged vectors;
                "q": list(qs[j % 2]),
                "k": ks[j],
                # identity V: output coordinate d accumulates ONLY w_d.
                "v": [1.0 if d == j else 0.0 for d in range(6)],
            }
        return pt, blocks

    def _manual_reduce(self, pt, blocks, q_idx):
        """Independent softmax reduction over the UNSCOPED subset only."""
        from src.kernels.reference_attention import roped_dot

        tags = ["global", "global", "local", "local", "local", "local"]
        owners = {0: None, 1: None, 2: (2, 4), 3: (2, 4), 4: (4, 6), 5: (4, 6)}

        def allowed(j):  # Table 1 spelled out by hand, no scoping_mask here
            if j == q_idx:
                return True
            if tags[q_idx] == "local":
                return tags[j] == "global" or owners[j] == owners[q_idx]
            return tags[j] == "global"

        keep = [j for j in range(6) if allowed(j)]
        scale = 1.0 / math.sqrt(6.0)
        scores = [
            roped_dot(blocks[pt[q_idx]]["q"], blocks[pt[j]]["k"], q_idx, j) * scale
            for j in keep
        ]
        w = _softmax(scores)
        want = [0.0] * 6
        for j, wij in zip(keep, w):
            want[j] += wij * blocks[pt[j]]["v"][j]
        return keep, want

    def test_masked_keys_zero_weight_and_output_equals_reduced_subset(self):
        pt, blocks = self._world()
        mask = block_sparse_mask(self.SPANS_6)
        for q_idx in range(6):
            got = golden_attention(pt, dict(blocks), q_idx, None, mask)
            keep, want = self._manual_reduce(pt, dict(blocks), q_idx)
            # Masked keys receive EXACTLY zero attention weight (exp(-inf)==0).
            for j in range(6):
                if j not in keep:
                    assert got[j] == 0.0, (q_idx, j, got[j])
            # Output equals the manual reduction of the unmasked subset...
            assert got == pytest.approx(want, rel=1e-12), (q_idx, got, want)
            # ...and weights still sum to one.
            assert sum(got) == pytest.approx(1.0, rel=1e-12)

    def test_scope_mask_rows_match_table1_asymmetry_end_to_end(self):
        """Spot-check the deliberate asymmetry through real attention math:
        LOCAL query 2 sees GLOBAL key 0 (weight > 0) while GLOBAL query 0
        gives LOCAL key 2 exactly zero."""
        pt, blocks = self._world()
        mask = block_sparse_mask(self.SPANS_6)
        assert mask[2][0] is True and mask[0][2] is False
        out_q2 = golden_attention(pt, dict(blocks), 2, None, mask)
        out_q0 = golden_attention(pt, dict(blocks), 0, None, mask)
        assert out_q2[0] > 0.0          # body sees signature
        assert out_q0[2] == 0.0         # global KV insulated from locals
