"""Tests for the TRUE multi-token-block golden oracle (Ouroboros v7.1).

These pin :func:`src.kernels.reference_attention.build_multi_token_payloads`
and :func:`src.kernels.reference_attention.golden_attention_mt` — the upgrade
that removes the reference's biggest documented simplification ("one token
per block"): payloads now hold ``BLOCK_SIZE == 64`` distinct token vectors per
physical block, and the oracle performs the mathematically exact dense
attention over the WHOLE chain (softmax across all valid keys of the
sequence, never per block), with rope at ABSOLUTE positions
``block_idx * 64 + slot`` and partially-filled tail blocks respected via
``seq_lens``.

Pinned behaviours:

* single block + single token reproduces the legacy ``golden_attention``
  result BIT-FOR-BIT (tables=None path reuses the identical primitives);
* two-block chain, d=2, small ints — hand-derived closed forms
  (score = cos/sin of the position gap / sqrt(d));
* ``seq_lens`` shorter than chain capacity: tail tokens carry ZERO weight
  mass (proved by bit-identical output under tail mutation);
* random 3x64 chains at d=8 match an independent dense re-computation over
  concatenated [Q|K|V] tensors within 1e-9 relative, outputs finite,
  softmax mass 1 over exactly the valid key set.

No torch / numpy / triton anywhere: stdlib only, runs on any host.
"""

import math

import pytest

from src.block_table import BLOCK_SIZE, PAD_ID, BlockTable
from src.kernels.reference_attention import (
    build_multi_token_payloads,
    golden_attention,
    golden_attention_mt,
    reference_from_block_table,
)


# ---------------------------------------------------------------------------
# Independent helpers — deliberately NOT calling the library under test.
# ---------------------------------------------------------------------------


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _rot2(v, angle):
    """Explicit 2-d rotation by radians — spelled out, not rope_rotate."""
    c, s = math.cos(angle), math.sin(angle)
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]


def _rot_generic(v, pos, base):
    """Interleaved-pair RoPE at absolute position ``pos`` for any even d."""
    d = len(v)
    out = []
    for t in range(d // 2):
        ang = pos * base ** (-2.0 * t / d)
        c, s = math.cos(ang), math.sin(ang)
        x0, x1 = v[2 * t], v[2 * t + 1]
        out.append(x0 * c - x1 * s)
        out.append(x0 * s + x1 * c)
    return out


def _softmax(xs):
    m = max(xs)
    ws = [math.exp(x - m) for x in xs]
    tot = sum(ws)
    return ws, tot


def _dense_reference(
    q_rows,
    k_rows,
    v_rows,
    q_positions,
    key_positions,
    base=10000.0,
    bias=None,
    scope=None,
):
    """Straightforward dense re-computation over concatenated [Q|K|V] tensors.

    One output vector per query position; also returns the pre-normalization
    softmax mass per query (must sit at 1 up to float rounding) — this is the
    independent oracle the MT golden is diffed against.
    """
    d = len(k_rows[0])
    scale = 1.0 / math.sqrt(d)
    outs, masses = [], []
    for qi, qpos in enumerate(q_positions):
        qr = _rot_generic(q_rows[qi], qpos, base)
        scores = []
        for ki, kpos in enumerate(key_positions):
            s = _dot(qr, _rot_generic(k_rows[ki], kpos, base)) * scale
            if bias is not None:
                s += float(bias[qpos][kpos])
            if scope is not None and not scope[qpos][kpos]:
                s = float("-inf")
            scores.append(s)
        ws, tot = _softmax(scores)
        masses.append(sum(ws) / tot)
        acc = [0.0] * d
        for w, v in zip(ws, v_rows):
            if w == 0.0:
                continue
            for dd in range(d):
                acc[dd] += w * v[dd]
        outs.append([x / tot for x in acc])
    return outs, masses


def _flatten_chain(mt_blocks, phys_ids, n_tokens, d_head):
    """Concatenate per-block token payloads into dense Q/K/V row lists."""
    qs, ks, vs = [], [], []
    left = n_tokens
    for b_idx, pid in enumerate(phys_ids):
        take = min(BLOCK_SIZE, left)
        for slot in range(take):
            qs.append(mt_blocks[pid]["q"][slot])
            ks.append(mt_blocks[pid]["k"][slot])
            vs.append(mt_blocks[pid]["v"][slot])
        left -= take
    assert left == 0
    return qs, ks, vs


# ---------------------------------------------------------------------------
# 1. Single block, single token == legacy golden_attention, bit-for-bit
# ---------------------------------------------------------------------------


class TestSingleBlockMatchesLegacyExactly:
    def test_mt_form_matches_legacy_bit_for_bit(self):
        bt = BlockTable(max_blocks=8)
        head = bt.allocate_chain(1)
        pt_list, legacy_blocks = reference_from_block_table(bt, head, 4)
        assert pt_list == [head]
        q = legacy_blocks[head]["q"]
        k = legacy_blocks[head]["k"]
        v = legacy_blocks[head]["v"]
        mt_blocks = {head: {"q": [list(q)], "k": [list(k)], "v": [list(v)]}}

        legacy = golden_attention(pt_list, legacy_blocks, 0, None, None)
        got = golden_attention_mt([pt_list], mt_blocks, [1], None, None)
        assert got == [legacy]  # exact float equality, no tolerance

    def test_legacy_flat_payload_compat(self):
        """MT oracle also accepts legacy flat (one-token) payloads directly."""
        bt = BlockTable(max_blocks=8)
        head = bt.allocate_chain(1)
        _, legacy_blocks = reference_from_block_table(bt, head, 6)
        legacy = golden_attention([head], legacy_blocks, 0, None, None)
        got = golden_attention_mt([[head]], legacy_blocks, [1], None, None)
        assert got == [legacy]

    def test_builder_payload_query_at_zero_matches_its_own_dense(self):
        """Builder payloads at d=4, query pos 0: rope-free dense agreement."""
        bt = BlockTable(max_blocks=8)
        head = bt.allocate_chain(1)
        mt_blocks = build_multi_token_payloads(bt, head, 4, seed="legacy-bridge")
        q = mt_blocks[head]["q"][0]
        ks = mt_blocks[head]["k"]
        vs = mt_blocks[head]["v"]
        want, _ = _dense_reference([q], ks, vs, [0], list(range(BLOCK_SIZE)))
        got = golden_attention_mt(
            [[head]], mt_blocks, [BLOCK_SIZE], None, None, query_positions=[0]
        )
        assert got[0] == pytest.approx(want[0], rel=1e-12, abs=1e-14)


# ---------------------------------------------------------------------------
# 2. Two-block chain, d=2, small ints — hand-computed closed forms
# ---------------------------------------------------------------------------


class TestTwoBlockHandComputed:
    def _world(self):
        """Chain [A, B], both blocks FULL (seq_len=128), d_head=2.

        Block A keys are all k=[1,0] -> rot_p(k) = (cos p, sin p), so the
        score against a query rotated to position i is cos(i - p) / sqrt(2).
        Block B keys are all k=[0,1] -> rot_p(k) = (-sin p, cos p), score
        sin(i - p) / sqrt(2). These closed forms ARE the hand computation.
        """
        a, b = 40, 41
        blocks = {
            a: {
                "q": [[1.0, 0.0] for _ in range(BLOCK_SIZE)],
                "k": [[1.0, 0.0] for _ in range(BLOCK_SIZE)],
                "v": [[float(s), float(BLOCK_SIZE - s)] for s in range(BLOCK_SIZE)],
            },
            b: {
                "q": [[1.0, 0.0] for _ in range(BLOCK_SIZE)],
                "k": [[0.0, 1.0] for _ in range(BLOCK_SIZE)],
                "v": [[float(s % 5), float(s % 3)] for s in range(BLOCK_SIZE)],
            },
        }
        return [a, b], blocks

    def test_last_token_full_chain_exact(self):
        pt, blocks = self._world()
        seq = 2 * BLOCK_SIZE
        got = golden_attention_mt([pt], blocks, [seq], None, None)
        # Query = token 127 (default): q=[1,0] rotated by 127 rad.
        keys = [
            (math.cos(127 - p)) / math.sqrt(2.0) for p in range(BLOCK_SIZE)
        ] + [(math.sin(127 - p)) / math.sqrt(2.0) for p in range(64, 128)]
        ws, tot = _softmax(keys)
        want = [0.0, 0.0]
        for w, (b_idx, slot) in zip(
            ws, [(0, s) for s in range(64)] + [(1, s) for s in range(64)]
        ):
            v = blocks[pt[b_idx]]["v"][slot]
            want[0] += w * v[0]
            want[1] += w * v[1]
        assert got[0][0] == pytest.approx(want[0] / tot, rel=1e-12)
        assert got[0][1] == pytest.approx(want[1] / tot, rel=1e-12)

    def test_mid_chain_query_via_query_positions(self):
        pt, blocks = self._world()
        seq = 2 * BLOCK_SIZE
        got = golden_attention_mt(
            [pt], blocks, [seq], None, None, query_positions=[2]
        )
        keys = [
            (math.cos(2 - p)) / math.sqrt(2.0) for p in range(BLOCK_SIZE)
        ] + [(math.sin(2 - p)) / math.sqrt(2.0) for p in range(64, 128)]
        ws, tot = _softmax(keys)
        want = [0.0, 0.0]
        for w, (b_idx, slot) in zip(
            ws, [(0, s) for s in range(64)] + [(1, s) for s in range(64)]
        ):
            v = blocks[pt[b_idx]]["v"][slot]
            want[0] += w * v[0]
            want[1] += w * v[1]
        assert got[0] == pytest.approx([want[0] / tot, want[1] / tot], rel=1e-12)

    def test_additive_bias_shifts_weights(self):
        pt, blocks = self._world()
        seq = 2 * BLOCK_SIZE
        bias = [
            [((i - j) % 7 - 3) * 0.125 for j in range(seq)] for i in range(seq)
        ]
        got = golden_attention_mt([pt], blocks, [seq], None, None, bias=bias)
        qp = seq - 1
        scores = []
        for p in range(BLOCK_SIZE):
            scores.append(math.cos(qp - p) / math.sqrt(2.0) + bias[qp][p])
        for p in range(64, seq):
            scores.append(math.sin(qp - p) / math.sqrt(2.0) + bias[qp][p])
        ws, tot = _softmax(scores)
        want = [0.0, 0.0]
        for w, (b_idx, slot) in zip(
            ws, [(0, s) for s in range(64)] + [(1, s) for s in range(64)]
        ):
            v = blocks[pt[b_idx]]["v"][slot]
            want[0] += w * v[0]
            want[1] += w * v[1]
        assert got[0] == pytest.approx([want[0] / tot, want[1] / tot], rel=1e-12)

    def test_scope_removes_columns_and_sentinel_v_leaks_nothing(self):
        pt, blocks = self._world()
        seq = 2 * BLOCK_SIZE
        poisoned = {
            pid: {
                comp: [list(vec) for vec in blocks[pid][comp]]
                for comp in ("q", "k", "v")
            }
            for pid in pt
        }
        banned = {0, 1, 63, 64, 100, 127}
        for gp in banned:
            b_idx, slot = divmod(gp, BLOCK_SIZE)
            poisoned[pt[b_idx]]["v"][slot] = [1.0e9, -1.0e9]
        scope = [[True] * seq for _ in range(seq)]
        for gp in banned:
            for i in range(seq):
                scope[i][gp] = False
        got = golden_attention_mt(
            [pt], poisoned, [seq], None, None, scope=scope, query_positions=[5]
        )
        scores = []
        for p in range(seq):
            base_s = (
                math.cos(5 - p) / math.sqrt(2.0)
                if p < BLOCK_SIZE
                else math.sin(5 - p) / math.sqrt(2.0)
            )
            scores.append(float("-inf") if p in banned else base_s)
        ws, tot = _softmax(scores)
        want = [0.0, 0.0]
        for w, (b_idx, slot) in zip(
            ws, [(0, s) for s in range(64)] + [(1, s) for s in range(64)]
        ):
            if w == 0.0:
                continue
            v = blocks[pt[b_idx]]["v"][slot]
            want[0] += w * v[0]
            want[1] += w * v[1]
        # 1e9 sentinels must contribute EXACTLY zero: rel 1e-15 proves it.
        assert got[0] == pytest.approx([want[0] / tot, want[1] / tot], rel=1e-15)

    def test_softmax_spans_the_whole_chain_not_per_block(self):
        """Per-block softmax would give a different (wrong) output vector."""
        pt, blocks = self._world()
        seq = 2 * BLOCK_SIZE
        got = golden_attention_mt([pt], blocks, [seq], None, None)
        # Wrong hypothesis: softmax independently inside each 64-key block.
        wrong = []
        for b_idx in range(2):
            scores = [
                (math.cos(127 - p) if b_idx == 0 else math.sin(127 - p))
                / math.sqrt(2.0)
                for p in range(b_idx * 64, (b_idx + 1) * 64)
            ]
            ws, tot = _softmax(scores)
            acc = [0.0, 0.0]
            for w, slot in zip(ws, range(64)):
                v = blocks[pt[b_idx]]["v"][slot]
                acc[0] += w * v[0]
                acc[1] += w * v[1]
            wrong.extend([x / tot for x in acc])
        combined_wrong = [
            0.5 * wrong[0] + 0.5 * wrong[2],
            0.5 * wrong[1] + 0.5 * wrong[3],
        ]
        assert got[0] != pytest.approx(combined_wrong, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. seq_lens shorter than chain capacity — tail tokens excluded exactly
# ---------------------------------------------------------------------------


class TestSeqLensTailExclusion:
    def _world(self, poison_value=12345.0):
        bt = BlockTable(max_blocks=16)
        head = bt.allocate_chain(2)
        row = list(bt.walk(head))
        assert len(row) == 2
        mt = build_multi_token_payloads(bt, row, 2, seed="tail")

        def copy():
            return {
                pid: {
                    comp: [list(vec) for vec in mt[pid][comp]]
                    for comp in ("q", "k", "v")
                }
                for pid in row
            }

        clean = copy()
        poisoned = copy()
        # Poison EVERYTHING from global token 70 onward (block 1, slots 6..63).
        b1 = row[1]
        for slot in range(6, BLOCK_SIZE):
            poisoned[b1]["q"][slot] = [9.0e6, -9.0e6]
            poisoned[b1]["k"][slot] = [1.0e6, 1.0e6]
            poisoned[b1]["v"][slot] = [poison_value, poison_value]
        return row, clean, poisoned

    def test_output_ignores_tail_poison_bit_for_bit(self):
        row, clean, poisoned = self._world()
        seq = 70  # block 0 full + 6 tokens of block 1
        got_clean = golden_attention_mt(
            [row], clean, [seq], None, None, query_positions=[69]
        )
        got_poison = golden_attention_mt(
            [row], poisoned, [seq], None, None, query_positions=[69]
        )
        # Bit-identical output => exactly ZERO weight mass on tail tokens.
        assert got_clean == got_poison
        qs, ks, vs = _flatten_chain(clean, row, seq, 2)
        want, _ = _dense_reference([qs[69]], ks, vs, [69], list(range(seq)))
        assert got_clean[0] == pytest.approx(want[0], rel=1e-12)

    def test_exactly_one_full_block_then_next_block_inert(self):
        row, clean, _ = self._world()
        b0, b1 = row
        # Poison ALL of block 1 (q/k/v); with seq_len=64 it must never load.
        fully_poisoned = {
            pid: {
                comp: [list(vec) for vec in clean[pid][comp]]
                for comp in ("q", "k", "v")
            }
            for pid in row
        }
        for slot in range(BLOCK_SIZE):
            fully_poisoned[b1]["q"][slot] = [9.0e6, -9.0e6]
            fully_poisoned[b1]["k"][slot] = [1.0e6, 1.0e6]
            fully_poisoned[b1]["v"][slot] = [777.0, 777.0]
        assert fully_poisoned[b0] == clean[b0]  # block 0 untouched
        seq = BLOCK_SIZE
        got_clean = golden_attention_mt(
            [row], clean, [seq], None, None, query_positions=[63]
        )
        got_poison = golden_attention_mt(
            [row], fully_poisoned, [seq], None, None, query_positions=[63]
        )
        assert got_clean == got_poison  # block 1 never touched

    def test_first_token_of_tail_block_participates(self):
        row, clean, _ = self._world()
        b0, b1 = row
        # seq_len=65 -> block 1 slot 0 IS valid; mutating it must change output.
        seq = BLOCK_SIZE + 1
        base_out = golden_attention_mt(
            [row], clean, [seq], None, None, query_positions=[64]
        )
        tweaked = {
            pid: {
                comp: [list(vec) for vec in clean[pid][comp]]
                for comp in ("q", "k", "v")
            }
            for pid in row
        }
        tweaked[b1]["k"][0] = [-3.0, 0.5]
        tweaked[b1]["v"][0] = [-30.0, 5.0]
        changed = golden_attention_mt(
            [row], tweaked, [seq], None, None, query_positions=[64]
        )
        assert base_out[0] != pytest.approx(changed[0], rel=1e-6)
        # ...while slots 1..63 of the tail block stay inert even when poisoned.
        still_inert = {
            pid: {
                comp: [list(vec) for vec in clean[pid][comp]]
                for comp in ("q", "k", "v")
            }
            for pid in row
        }
        still_inert[b1]["k"][1] = [1.0e6, 1.0e6]
        still_inert[b1]["v"][63] = [8888.0, 8888.0]
        got = golden_attention_mt(
            [row], still_inert, [seq], None, None, query_positions=[64]
        )
        want_qs, want_ks, want_vs = _flatten_chain(clean, row, seq, 2)
        want, _ = _dense_reference(
            [want_qs[64]], want_ks, want_vs, [64], list(range(seq))
        )
        assert got[0] == pytest.approx(want[0], rel=1e-12)

    def test_padded_row_beyond_seq_is_fine(self):
        row, clean, _ = self._world()
        padded = list(row) + [PAD_ID, PAD_ID, PAD_ID]
        got_pad = golden_attention_mt([padded], clean, [70], None, None)
        got_plain = golden_attention_mt([row], clean, [70], None, None)
        assert got_pad == got_plain

    def test_error_paths(self):
        row, clean, _ = self._world()
        with pytest.raises(RuntimeError, match="seq_len"):
            golden_attention_mt([row], clean, [0], None, None)
        with pytest.raises(RuntimeError, match="capacity"):
            golden_attention_mt([[row[0]]], clean, [129], None, None)
        with pytest.raises(RuntimeError, match="capacity"):
            golden_attention_mt([row], clean, [193], None, None)
        with pytest.raises(RuntimeError, match="seq_lens"):
            golden_attention_mt([row, row], clean, [70], None, None)
        with pytest.raises(RuntimeError, match="no block payload"):
            golden_attention_mt([[row[0], 99]], clean, [128], None, None)
        with pytest.raises(RuntimeError, match="query"):
            golden_attention_mt(
                [row], clean, [70], None, None, query_positions=[70]
            )
        with pytest.raises(RuntimeError, match="fully masked"):
            scope = [[False] * 70 for _ in range(70)]
            golden_attention_mt([row], clean, [70], None, None, scope=scope)


# ---------------------------------------------------------------------------
# 4. Random chains (3 blocks x 64 tokens, d=8): dense parity within 1e-9
# ---------------------------------------------------------------------------


class TestRandomChainsDenseParity:
    BASE = 777.0

    def _rope_tables(self, n_positions, d_head):
        cos_t, sin_t = [], []
        for p in range(n_positions):
            crow, srow = [], []
            for t in range(d_head // 2):
                ang = p * self.BASE ** (-2.0 * t / d_head)
                crow.append(math.cos(ang))
                srow.append(math.sin(ang))
            cos_t.append(crow)
            sin_t.append(srow)
        return cos_t, sin_t

    def test_full_and_partial_chains_match_dense_recomputation(self):
        bt = BlockTable(max_blocks=32)
        h_full = bt.allocate_chain(3)
        h_part = bt.allocate_chain(3)
        mt = build_multi_token_payloads(bt, [h_full, h_part], 8, seed="parity")

        cases = [(h_full, 3 * BLOCK_SIZE), (h_part, 130)]
        probes = {
            3 * BLOCK_SIZE: [0, 1, 62, 63, 64, 100, 191],
            130: [0, 64, 100, 129],
        }
        cos_t, sin_t = self._rope_tables(3 * BLOCK_SIZE, 8)
        for head, seq in cases:
            row = list(bt.walk(head))
            qs, ks, vs = _flatten_chain(mt, row, seq, 8)
            q_positions = probes[seq]
            want, masses = _dense_reference(
                [qs[p] for p in q_positions],
                ks,
                vs,
                q_positions,
                list(range(seq)),
                base=self.BASE,
            )
            got_analytic = golden_attention_mt(
                [row],
                mt,
                [seq],
                None,
                None,
                base=self.BASE,
                query_positions=q_positions,
            )
            got_table = golden_attention_mt(
                [row],
                mt,
                [seq],
                cos_t,
                sin_t,
                base=self.BASE,
                query_positions=q_positions,
            )
            assert len(got_analytic) == len(q_positions)
            for g_a, g_t, w in zip(got_analytic, got_table, want):
                # finite everywhere
                assert all(math.isfinite(x) for x in g_a)
                # analytic path == independent dense recompute, rel 1e-9
                assert g_a == pytest.approx(w, rel=1e-9, abs=1e-12)
                # prebuilt cos/sin tables agree with the analytic path
                assert g_t == pytest.approx(g_a, rel=1e-12, abs=1e-15)
            for m in masses:
                assert m == pytest.approx(1.0, abs=1e-12)

    def test_uniform_logits_put_mass_one_over_exactly_the_valid_keys(self):
        """Counter-rotated keys flatten every logit -> output == mean(v).

        Any mass leaking to invalid keys, or any normalization error, breaks
        the mean-of-v identity.
        """
        from src.kernels.reference_attention import rope_rotate

        bt = BlockTable(max_blocks=16)
        head = bt.allocate_chain(2)
        row = list(bt.walk(head))
        mt = build_multi_token_payloads(bt, row, 2, seed="uniform")
        seq = 70
        flat = {
            pid: {
                comp: [list(vec) for vec in mt[pid][comp]]
                for comp in ("q", "k", "v")
            }
            for pid in row
        }
        # Pre-counter-rotate every key: rot_p(k_p) == [1, 0] (up to float
        # rounding), so ALL logits collapse to the same value -> uniform
        # softmax -> output == mean of the valid v vectors, EXACTLY.
        for p in range(seq):
            b_idx, slot = divmod(p, BLOCK_SIZE)
            flat[row[b_idx]]["k"][slot] = rope_rotate([1.0, 0.0], -p)
            flat[row[b_idx]]["v"][slot] = [float(p) + 1.0, float(p) * 2.0]
        # Poison the invalid tail with absurd v: must not move the mean.
        for p in range(seq, len(row) * BLOCK_SIZE):
            b_idx, slot = divmod(p, BLOCK_SIZE)
            flat[row[b_idx]]["v"][slot] = [1.0e12, -1.0e12]
        got = golden_attention_mt(
            [row], flat, [seq], None, None, query_positions=[33]
        )
        want_x = sum(float(p) + 1.0 for p in range(seq)) / seq
        want_y = sum(float(p) * 2.0 for p in range(seq)) / seq
        assert got[0] == pytest.approx([want_x, want_y], rel=1e-12)


# ---------------------------------------------------------------------------
# 5. build_multi_token_payloads: determinism, shapes, validation
# ---------------------------------------------------------------------------


class TestBuildMultiTokenPayloads:
    def test_shapes_are_block_size_by_d_head_per_component(self):
        bt = BlockTable(max_blocks=8)
        head = bt.allocate_chain(2)
        mt = build_multi_token_payloads(bt, head, 8, seed="shapes")
        assert set(mt.keys()) == set(bt.walk(head))
        for pid in mt:
            for comp in ("q", "k", "v"):
                assert len(mt[pid][comp]) == BLOCK_SIZE
                for vec in mt[pid][comp]:
                    assert len(vec) == 8
                    assert all(isinstance(x, float) for x in vec)
                    assert all(-1.0 <= x <= 1.0 for x in vec)

    def test_deterministic_same_seed_and_distinct_slots(self):
        bt = BlockTable(max_blocks=8)
        head = bt.allocate_chain(1)
        a = build_multi_token_payloads(bt, head, 4, seed="det")
        b = build_multi_token_payloads(bt, head, 4, seed="det")
        assert a == b  # deep equality across calls/processes
        c = build_multi_token_payloads(bt, head, 4, seed="other")
        assert a != c
        pid = head
        slots = {tuple(vec) for vec in a[pid]["k"]}
        assert len(slots) == BLOCK_SIZE  # all 64 slot vectors distinct
        assert a[pid]["q"][0] != a[pid]["k"][0] != a[pid]["v"][0]

    def test_head_list_input_equals_single_heads_union(self):
        bt = BlockTable(max_blocks=16)
        h1 = bt.allocate_chain(2)
        h2 = bt.allocate_chain(1)
        both = build_multi_token_payloads(bt, [h1, h2], 4, seed="union")
        one = build_multi_token_payloads(bt, h1, 4, seed="union")
        assert both[h1] == one[h1]
        assert set(both) == {*bt.walk(h1), *bt.walk(h2)}

    def test_validation_errors(self):
        bt = BlockTable(max_blocks=8)
        head = bt.allocate_chain(1)
        with pytest.raises(RuntimeError, match="d_head"):
            build_multi_token_payloads(bt, head, 3, seed="s")
        with pytest.raises(RuntimeError, match="d_head"):
            build_multi_token_payloads(bt, head, 1, seed="s")
        with pytest.raises(RuntimeError, match="allocated"):
            build_multi_token_payloads(bt, head + 5, 4, seed="s")
        with pytest.raises(RuntimeError, match="head"):
            build_multi_token_payloads(bt, [], 4, seed="s")
