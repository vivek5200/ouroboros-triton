"""Pure-python golden reference for block attention (paper §4.3).

This module is the stdlib-only oracle the Triton kernel in
:mod:`src.kernels.block_attention` must match bit-for-bounded. It contains
NO torch and NO numpy — every operation is plain Python floats, so the
semantics are readable line-by-line and testable on any host.

Per query token ``i`` over a chain of ``n`` key blocks it performs exactly
the §4.3 pipeline:

1. walk the page table to resolve logical slot -> physical block id;
2. load that block's q/k/v vectors;
3. apply the 1-D RoPE rotation to q/k at their *logical* positions;
4. score ``= dot(q_rot, k_rot) / sqrt(d)``;
5. ADD the AST graph bias pre-softmax (law math-rope);
6. apply the block-sparse scoping mask (False -> ``-inf`` logit);
7. softmax, then the weighted sum of v vectors.

DOCUMENTED SIMPLIFICATIONS (reference vs. engine):

* **Page table form** — the engine's page table is a padded 2-D row
  (``BlockTable.page_table_row(head, pad_to=...)``, PAD_ID-filled) that the
  kernel chases as a linked list on device via ``PT + seq*stride_pts +
  n*stride_ptb`` loads. Here a chain is simply an ordered ``list[int]`` of
  physical block ids in walk order — exactly what
  ``list(bt.walk(head))`` / an unpadded ``page_table_row`` produces, so the
  adapter below hands back the real ids with no translation.
* **One token per block** — the engine stores 64 tokens per physical block
  (``src.block_table.BLOCK_SIZE``); this reference keeps ONE token vector of
  length ``d_head`` per physical block, so "logical position" == chain index
  and token-granular bias/mask degenerate to block granularity (the kernel's
  per-(q-block, kv-block) scope mask becomes a per-(i, j) flag). Extending
  to BLOCK_SIZE tokens per block is a mechanical repeat of steps 2-7 per
  token offset; it is deliberately left out to keep the oracle auditable.
* **RoPE layout** — interleaved complex pairs: ``(x[2t], x[2t+1])`` is
  rotated by ``pos * base**(-2t/d)`` (θ_i = base^(-2i/d)). The Triton kernel
  applies the same rotations to the two feature *halves*
  (``(x[t], x[t+D/2])``); matching against the kernel therefore requires the
  fixed channel permutation ``perm(t) = (t%2)*HALF + t//2`` on both sides —
  same math, different memory interleave. With ``d_head == 2`` (the tests'
  regime) both layouts coincide exactly.
"""

import math
import random

from src.block_table import BlockTable

__all__ = [
    "rope_rotate",
    "roped_dot",
    "golden_attention",
    "reference_from_block_table",
]


def _require_even_vector(vector, name: str) -> int:
    try:
        n = len(vector)
    except TypeError:
        raise RuntimeError(f"{name} must be a sequence, got {type(vector).__name__}")
    if n % 2 != 0:
        raise RuntimeError(
            f"{name} must have even length for pair-wise RoPE, got {n}"
        )
    return n


def rope_rotate(vector, position: int, base: float = 10000.0):
    """Pair-wise complex RoPE rotation of ``vector`` at ``position``.

    Every consecutive pair ``(x[2t], x[2t+1])`` is rotated by the angle
    ``position * theta_t`` with ``theta_t = base**(-2*t/d)`` (``d`` = full
    vector length), i.e. multiplication by the unit complex number
    ``e^{i * position * theta_t}``:

        (x0, x1) -> (x0*c - x1*s, x0*s + x1*c)

    Returns a NEW list; the input is untouched. With ``d == 2`` there is a
    single pair and ``theta_0 == base**0 == 1`` rad per position step.
    """
    d = _require_even_vector(vector, "vector")
    if isinstance(position, bool) or not isinstance(position, int):
        raise RuntimeError(f"invalid rope position: {position!r}")
    if not isinstance(base, (int, float)) or base <= 0.0:
        raise RuntimeError(f"rope base must be > 0, got {base!r}")
    out: list[float] = []
    for t in range(d // 2):
        theta = float(base) ** (-2.0 * t / d)
        angle = float(position) * theta
        c, s = math.cos(angle), math.sin(angle)
        x0, x1 = float(vector[2 * t]), float(vector[2 * t + 1])
        out.append(x0 * c - x1 * s)
        out.append(x0 * s + x1 * c)
    return out


def roped_dot(q, k, q_pos: int, k_pos: int, base: float = 10000.0) -> float:
    """``dot(rope(q, q_pos), rope(k, k_pos))`` — the shared score primitive.

    Exposed because the relative-offset law (Module 3) is stated on scores:
    for fixed q/k contents the value depends only on the gap ``q_pos -
    k_pos``. :func:`golden_attention` uses exactly this primitive, so tests
    can diff kernel scores against one audited function.
    """
    qr = rope_rotate(q, q_pos, base)
    kr = rope_rotate(k, k_pos, base)
    return sum(a * b for a, b in zip(qr, kr))


def golden_attention(
    page_table,
    blocks,
    q_idx: int,
    bias,
    scope_mask,
    position_base: float = 10000.0,
):
    """Golden output vector for query token ``q_idx`` over one block chain.

    Args:
        page_table: ordered list of physical block ids (chain order). See the
            module docstring for the mapping from the engine's linked-list /
            padded-row page table.
        blocks: ``{block_id: {"q": [...], "k": [...], "v": [...]}}`` — flat
            float vectors of length ``d_head`` (one token per block).
        q_idx: logical index of the query token within the chain (its rope
            position is also ``q_idx``).
        bias: additive AST graph bias, ``bias[i][j]`` added to score(i, j)
            PRE-softmax, or ``None``.
        scope_mask: block-sparse scoping flags, ``scope_mask[i][j]]``
            truthy = allowed, falsy = key j removed (``-inf`` logit), or
            ``None`` (= all True). Kernel masks are int8 0/≠0 — any truthy
            value is accepted here under the same contract.
        position_base: RoPE base (default 10000.0, law math-rope).

    Returns:
        ``list[float]`` of length ``d_head`` — the attention output for
        query ``q_idx``.

    Raises:
        RuntimeError: on an empty chain, a page-table entry with no backing
            block payload, an out-of-range ``q_idx``, mismatched head dims,
            or a query whose every key column is masked out (the kernel's
            online softmax would divide by a zero running sum).
    """
    n = len(page_table)
    if n == 0:
        raise RuntimeError("golden_attention: page-table chain is empty")
    if isinstance(q_idx, bool) or not isinstance(q_idx, int):
        raise RuntimeError(f"invalid q_idx: {q_idx!r}")
    if q_idx < 0 or q_idx >= n:
        raise RuntimeError(f"q_idx {q_idx} out of range for chain of {n} block(s)")

    # ---- (1) page-table walk + load ----------------------------------------
    resolved = []
    for logical, phys in enumerate(page_table):
        try:
            present = phys in blocks
        except TypeError:
            present = False
        if not present:
            raise RuntimeError(
                f"page table entry {logical} -> physical block {phys!r} has no "
                f"block payload"
            )
        resolved.append(blocks[phys])

    d_head = _require_even_vector(resolved[q_idx]["q"], "query vector")
    scale = 1.0 / math.sqrt(float(d_head))

    # ---- (2)-(5): rope, dot, additive bias ---------------------------------
    scores: list[float] = []
    for j, payload in enumerate(resolved):
        k_vec = payload["k"]
        if len(k_vec) != d_head:
            raise RuntimeError(
                f"key vector at logical slot {j} has dim {len(k_vec)}, expected {d_head}"
            )
        s = roped_dot(resolved[q_idx]["q"], k_vec, q_idx, j, position_base) * scale
        if bias is not None:
            s += float(bias[q_idx][j])
        scores.append(s)

    # ---- (6) block-sparse scoping mask --------------------------------------
    if scope_mask is not None:
        for j in range(n):
            if not scope_mask[q_idx][j]:
                scores[j] = float("-inf")
    if all(s == float("-inf") for s in scores):
        raise RuntimeError(
            f"query {q_idx} is fully masked: no key contributes weight"
        )

    # ---- (7) stable softmax + weighted sum of V -----------------------------
    m = max(scores)
    weights = [math.exp(s - m) for s in scores]
    total = sum(weights)

    out = [0.0] * d_head
    for j, payload in enumerate(resolved):
        v_vec = payload["v"]
        if len(v_vec) != d_head:
            raise RuntimeError(
                f"value vector at logical slot {j} has dim {len(v_vec)}, "
                f"expected {d_head}"
            )
        w = weights[j]
        if w == 0.0:
            continue  # masked column contributes nothing, skip the FMA noise
        for dd in range(d_head):
            out[dd] += w * float(v_vec[dd])
    return [x / total for x in out]


# ---------------------------------------------------------------------------
# Adapter: live BlockTable chains -> golden-reference inputs
# ---------------------------------------------------------------------------


def reference_from_block_table(bt: BlockTable, head: int, d_head: int):
    """Materialize ``(page_table_list, blocks)`` from a live chain.

    The page table is the REAL chain order — ``list(bt.walk(head))``, i.e.
    the unpadded form of ``bt.page_table_row(head)`` — so future kernel
    tests can diff against actual BlockTable state with zero translation.

    Payload vectors are deterministic stand-ins seeded by physical id
    (``random.Random`` with a string seed → sha512-based, stable across
    processes/platforms). The engine cache holds ``[n_phys, H, 64, D]``
    tensors; the reference needs only one length-``d_head`` q/k/v triple per
    physical block, keyed by block id, so identity/ordering — what these
    tests verify — is preserved exactly.

    Raises:
        RuntimeError: if ``head`` is not an allocated block or ``d_head``
            is not even and >= 2.
    """
    if isinstance(d_head, bool) or not isinstance(d_head, int) or d_head < 2 or d_head % 2 != 0:
        raise RuntimeError(f"d_head must be an even integer >= 2, got {d_head!r}")
    page_table_list = list(bt.walk(head))  # raises on unallocated head
    components = ("q", "k", "v")
    blocks: dict[int, dict[str, list[float]]] = {}
    for pid in page_table_list:
        payload: dict[str, list[float]] = {}
        for comp in components:
            rng = random.Random(f"ouroboros-ref:{comp}:{pid}:{d_head}")
            payload[comp] = [rng.uniform(-1.0, 1.0) for _ in range(d_head)]
        blocks[pid] = payload
    return page_table_list, blocks
