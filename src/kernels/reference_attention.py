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
  token offset; it is deliberately left out of :func:`golden_attention` to
  keep that oracle auditable.

  **LIFTED in v7.1** — :func:`build_multi_token_payloads` and
  :func:`golden_attention_mt` below implement TRUE multi-token blocks: 64
  distinct q/k/v vectors per physical block, RoPE at ABSOLUTE positions
  ``block_idx * BLOCK_SIZE + slot``, partially-filled tail blocks via
  ``seq_lens``, and a stable softmax over ALL valid keys of the sequence
  (across the whole chain, never per block). That function is the
  mathematically exact dense attention over the chain — THE oracle the
  Triton kernel must match at ANY even ``d_head`` once the channel
  permutation (:func:`apply_channel_perm`) is applied on both sides.
* **RoPE layout** — interleaved complex pairs: ``(x[2t], x[2t+1])`` is
  rotated by ``pos * base**(-2t/d)`` (θ_i = base^(-2i/d)). The Triton kernel
  applies the same rotations to the two feature *halves*
  (``(x[t], x[t+D/2])``); matching against the kernel therefore requires the
  fixed channel permutation ``perm(t) = (t%2)*HALF + t//2`` on both sides —
  same math, different memory interleave. With ``d_head == 2`` (the tests'
  regime) both layouts coincide exactly.

  That bridge is implemented by :func:`perm`, :func:`apply_channel_perm` and
  :func:`invert_channel_perm` below. Mathematical note (verified): the
  permutation is an involution ONLY for ``d_head in {2, 4}`` — its order is
  the multiplicative order of 2 mod ``d_head - 1``, which is 3 at
  ``d_head == 8`` (cycles ``(1 4 2)(3 5 6)``, 0 and 7 fixed) — so round-trips
  must go through the explicit inverse, never the forward map.
"""

import hashlib
import math
import random

from src.block_table import BLOCK_SIZE, PAD_ID, BlockTable

__all__ = [
    "rope_rotate",
    "roped_dot",
    "perm",
    "apply_channel_perm",
    "invert_channel_perm",
    "golden_attention",
    "reference_from_block_table",
    "build_multi_token_payloads",
    "golden_attention_mt",
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


# ---------------------------------------------------------------------------
# Channel-permutation bridge: interleaved layout (this oracle) <-> half-split
# layout (the Triton kernel). See "RoPE layout" in the module docstring.
# ---------------------------------------------------------------------------


def perm(t: int, half: int) -> int:
    """Map interleaved-pair channel ``t`` to its half-split slot.

    The documented relabeling (module docstring, "RoPE layout")::

        perm(t) = (t % 2) * half + t // 2

    Interleaved pair ``j`` is ``(x[2j], x[2j+1])``; half-split pair ``j`` is
    ``(x[j], x[j + half])``. Both are rotated by the SAME angle
    ``base**(-2j/d)``, so the map is pure layout: first components
    ``t = 2j -> j``, second components ``t = 2j + 1 -> j + half``. The inverse
    mapping is ``perm_inv(m) = 2*m`` for ``m < half`` else ``2*(m - half) + 1``.

    INVOLUTION STATUS (verified mathematically): ``perm`` is the classic
    out-shuffle on ``d = 2 * half`` slots, so its order equals the
    multiplicative order of 2 modulo ``d - 1`` (well defined since even ``d``
    gives ``gcd(2, d - 1) = 1``). It is therefore its OWN inverse only for
    ``d_head in {2, 4}`` (where ``(d - 1) | 3``); at ``d_head == 8`` the order
    is 3 — cycles ``(1 4 2)(3 5 6)`` with channels 0 and 7 fixed — so there
    the inverse DIFFERS from the forward permutation.

    Raises:
        RuntimeError: if ``t`` is not an in-domain int or ``half`` invalid.
    """
    if isinstance(t, bool) or not isinstance(t, int):
        raise RuntimeError(f"invalid channel index: {t!r}")
    if isinstance(half, bool) or not isinstance(half, int) or half <= 0:
        raise RuntimeError(f"invalid half width: {half!r}")
    if t < 0 or t >= 2 * half:
        raise RuntimeError(f"channel index {t} out of range for d_head={2 * half}")
    return (t % 2) * half + t // 2


def _require_matching_half(vector, half: int) -> None:
    d = _require_even_vector(vector, "vector")
    if isinstance(half, bool) or not isinstance(half, int) or 2 * half != d:
        raise RuntimeError(
            f"half must satisfy 2*half == len(vector) == {d}, got {half!r}"
        )


def apply_channel_perm(vec, half: int):
    """Relabel ``vec`` from interleaved-pair layout to kernel half-split layout.

    ``out[perm(t, half)] == vec[t]`` for every channel ``t``. Applying this to
    a golden-reference quantity (q/k/v payload vectors AND attention output
    rows, which inherit v's layout) makes it directly comparable to the
    kernel's half-split convention at any even ``d_head == 2 * half``; it is
    the identity at ``d_head == 2``. Rotation commutes with the map,
    ``R_halfsplit(perm(x)) == perm(R_interleaved(x))``, so scores computed on
    permuted inputs equal interleaved-layout scores exactly.

    Returns a NEW list; the input is untouched.
    """
    _require_matching_half(vec, half)
    out: list[float] = [0.0] * len(vec)
    for t in range(len(vec)):
        out[perm(t, half)] = vec[t]
    return out


def invert_channel_perm(vec, half: int):
    """Exact inverse of :func:`apply_channel_perm`: ``out[t] == vec[perm(t, half)]``.

    NOTE: the inverse coincides with the forward permutation ONLY when it is
    an involution, i.e. for ``d_head = 2 * half in {2, 4}`` (see
    :func:`perm`); NOT at e.g. ``d_head == 8``, where the forward map has
    order 3. Always route round-trips through this function.

    Returns a NEW list; the input is untouched.
    """
    _require_matching_half(vec, half)
    return [vec[perm(t, half)] for t in range(len(vec))]


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


# ---------------------------------------------------------------------------
# TRUE multi-token blocks (v7.1): deterministic 64-token payloads and the
# exact dense-attention oracle over a whole block chain. See the module
# docstring, simplification bullet #2 ("LIFTED in v7.1").
# ---------------------------------------------------------------------------

_MT_COMPONENTS = ("q", "k", "v")
_TWO64 = float(1 << 64)


def _sha512_uniform(tag: str, count: int) -> list[float]:
    """``count`` deterministic floats in [-1, 1) hashed from ``tag``.

    Same spirit as ``reference_from_block_table``'s string-seeded
    ``random.Random`` (CPython seeds it via sha512 of the string) but fully
    explicit and position-addressable: digest counter ``c`` is
    ``sha512(f"{tag}:{c}")``, split into big-endian 8-byte words mapped to
    ``u / 2^64 * 2 - 1``. Stable across processes/platforms/Python versions,
    and each (block id, slot, component) draws from an independent stream.
    """
    if count <= 0:
        raise RuntimeError(f"count must be >= 1, got {count!r}")
    vals: list[float] = []
    counter = 0
    while len(vals) < count:
        digest = hashlib.sha512(f"{tag}:{counter}".encode("utf-8")).digest()
        for off in range(0, len(digest) - 7, 8):
            u = int.from_bytes(digest[off : off + 8], "big")
            vals.append(u / _TWO64 * 2.0 - 1.0)
            if len(vals) == count:
                break
        counter += 1
    return vals


def build_multi_token_payloads(bt: BlockTable, head_chain_ids, d_head: int, seed):
    """Deterministic q/k/v payloads with BLOCK_SIZE=64 tokens per block.

    This is the multi-token counterpart of :func:`reference_from_block_table`:
    instead of ONE length-``d_head`` vector per physical block it generates
    ``BLOCK_SIZE`` DISTINCT token vectors per component per physical block —
    exactly the ``[n_phys, H, 64, D]`` shape the engine cache holds.

    Args:
        bt: live :class:`src.block_table.BlockTable` (chains are read via
            ``bt.walk(head)``, so ids are REAL allocated block ids).
        head_chain_ids: one chain head (int) or an iterable of chain heads;
            the result is the union of every physical block on those chains
            (shared blocks dedup to one entry).
        d_head: even head dim >= 2.
        seed: str or int; tags the hash streams. Vectors depend ONLY on
            ``(seed, component, block_id, slot, d_head)`` via sha512 —
            identical inputs reproduce identical payloads anywhere.

    Returns:
        ``{block_id: {"q": [64][d], "k": [64][d], "v": [64][d]}}`` — slot
        index within a block == logical token offset inside that block, so
        absolute rope position = ``block_idx * BLOCK_SIZE + slot``.

    Raises:
        RuntimeError: invalid ``d_head``/``seed``, no heads given, a
            non-int head entry, or an unallocated head (via ``walk``).
    """
    if (
        isinstance(d_head, bool)
        or not isinstance(d_head, int)
        or d_head < 2
        or d_head % 2 != 0
    ):
        raise RuntimeError(f"d_head must be an even integer >= 2, got {d_head!r}")
    if isinstance(seed, bool) or not isinstance(seed, (str, int)):
        raise RuntimeError(f"seed must be a str or int, got {type(seed).__name__}")
    if isinstance(head_chain_ids, bool):
        raise RuntimeError(f"invalid chain head: {head_chain_ids!r}")
    if isinstance(head_chain_ids, int):
        heads = [head_chain_ids]
    else:
        if isinstance(head_chain_ids, str):
            raise RuntimeError(
                f"head_chain_ids must be an int head or an iterable of ints, "
                f"got a str"
            )
        try:
            heads = list(head_chain_ids)
        except TypeError:
            raise RuntimeError(
                f"head_chain_ids must be an int head or an iterable of ints, "
                f"got {type(head_chain_ids).__name__}"
            )
        if not heads:
            raise RuntimeError("head_chain_ids contains no chain heads")
        for h in heads:
            if isinstance(h, bool) or not isinstance(h, int):
                raise RuntimeError(f"invalid chain head: {h!r}")

    tag_seed = str(seed)
    blocks: dict[int, dict[str, list[list[float]]]] = {}
    for head in heads:
        for pid in bt.walk(head):  # raises on unallocated head
            if pid in blocks:
                continue
            payload: dict[str, list[list[float]]] = {}
            for comp in _MT_COMPONENTS:
                payload[comp] = [
                    _sha512_uniform(
                        f"ouroboros-mt:{tag_seed}:{comp}:{pid}:{slot}:{d_head}",
                        d_head,
                    )
                    for slot in range(BLOCK_SIZE)
                ]
            blocks[pid] = payload
    return blocks


def _mt_block_tokens(payload, pid: int):
    """Adapt one block payload into per-component lists of token vectors.

    Accepts BOTH payload shapes:
      * multi-token (from :func:`build_multi_token_payloads`):
        ``payload[c] == [[...], [...], ...]`` — used as-is;
      * legacy flat (from :func:`reference_from_block_table`):
        ``payload[c] == [...]`` — treated as a single token at slot 0.
    Returns ``(toks, n_stored)`` where ``toks[c][s]`` is token ``s``'s vector.
    """
    toks: dict[str, list[list[float]]] = {}
    n_stored: int | None = None
    for comp in _MT_COMPONENTS:
        try:
            arr = payload[comp]
        except (TypeError, KeyError):
            raise RuntimeError(
                f"physical block {pid!r} payload has no '{comp}' component"
            )
        if isinstance(arr, (list, tuple)) and arr and isinstance(arr[0], (list, tuple)):
            vecs = [[float(x) for x in vec] for vec in arr]
        else:
            try:
                vecs = [[float(x) for x in arr]]
            except TypeError:
                raise RuntimeError(
                    f"physical block {pid!r} has malformed '{comp}' payload"
                )
            if len(vecs[0]) == 0 and len(arr) == 0:
                raise RuntimeError(
                    f"physical block {pid!r} has empty '{comp}' payload"
                )
        toks[comp] = vecs
        if n_stored is None:
            n_stored = len(vecs)
        elif len(vecs) != n_stored:
            raise RuntimeError(
                f"physical block {pid!r} stores inconsistent token counts "
                f"across q/k/v ({n_stored} vs {len(vecs)})"
            )
    return toks, n_stored


def _rope_at(vector, position: int, cos_table, sin_table, base: float, name: str):
    """RoPE at ABSOLUTE ``position`` using prebuilt tables or analytic angles.

    With both tables ``None`` this IS :func:`rope_rotate` (bit-identical call
    path). Otherwise ``cos_table[p][t] / sin_table[p][t]`` supply cos/sin of
    position ``p``'s angle for frequency lane ``t`` (lane count must cover
    ``d // 2``), letting the oracle consume the exact rope table tensor the
    kernel indexes.
    """
    d = _require_even_vector(vector, name)
    if cos_table is None and sin_table is None:
        return rope_rotate(vector, position, base)
    if cos_table is None or sin_table is None:
        raise RuntimeError("cos_table and sin_table must be provided together")
    if isinstance(position, bool) or not isinstance(position, int):
        raise RuntimeError(f"invalid rope position: {position!r}")
    try:
        c_row = cos_table[position]
        s_row = sin_table[position]
    except IndexError:
        raise RuntimeError(
            f"rope position {position} outside rope tables "
            f"({len(cos_table)} entries)"
        )
    except TypeError:
        raise RuntimeError("cos_table/sin_table must be sequences indexed by position")
    half = d // 2
    try:
        lanes = [(float(c_row[t]), float(s_row[t])) for t in range(half)]
    except IndexError:
        raise RuntimeError(
            f"rope table row {position} shorter than required {half} lane(s)"
        )
    out: list[float] = []
    for t in range(half):
        c, s = lanes[t]
        x0, x1 = float(vector[2 * t]), float(vector[2 * t + 1])
        out.append(x0 * c - x1 * s)
        out.append(x0 * s + x1 * c)
    return out


def golden_attention_mt(
    page_table_rows,
    blocks,
    seq_lens,
    cos_table,
    sin_table,
    bias=None,
    scope=None,
    base: float = 10000.0,
    query_positions=None,
):
    """EXACT dense attention over TRUE multi-token block chains (v7.1 oracle).

    The mathematically exact §4.3 pipeline at token granularity — the oracle
    the Triton kernel must match at ANY even ``d_head`` once the channel
    permutation (:func:`apply_channel_perm` on payloads and output rows) is
    applied to both sides. Per queried token ``i`` of a sequence:

    1. walk the row's VALID page-table entries (``PAD_ID`` skipped);
    2. load each block's 64 token vectors (legacy flat payloads accepted as
       one-token blocks — see :func:`_mt_block_tokens`);
    3. rope q/k by ABSOLUTE positions ``block_idx * BLOCK_SIZE + slot``
       (tables or analytic angles — see :func:`_rope_at`);
    4. score ``= dot(q_rot, k_rot) / sqrt(d)``;
    5. ADD ``bias[i][j]`` PRE-softmax if given;
    6. scoping mask ``scope[i][j]`` (truthy = allowed; None = all-ones);
    7. stable softmax over ALL valid keys of the SEQUENCE — across the whole
       chain, never per block — then the weighted sum of v.

    Tail handling: only ``seq_len - full_blocks * BLOCK_SIZE`` leading slots
    of the last participating block take part, so weight mass sits exclusively
    on valid positions.

    Args:
        page_table_rows: one row PER (sequence, head) pair; a row is the
            ordered list of physical block ids in chain order (the unpadded
            form of ``BlockTable.page_table_row``; PAD_ID tails tolerated).
        blocks: ``{block_id: {"q"/"k"/"v": [token][d]}}`` from
            :func:`build_multi_token_payloads` (or legacy flat vectors).
        seq_lens: one integer >= 1 per row; tokens beyond it are ignored.
        cos_table / sin_table: optional-but-positional rope tables indexed
            ``[absolute_position][frequency_lane]`` built under the same
            ``base``; pass ``None`` for both to rotate analytically (the
            bit-identical legacy path).
        bias: additive AST bias shared across rows, ``bias[i][j]`` with i/j
            GLOBAL token indices inside the sequence, or None. Callers needing
            per-row bias/mask should invoke once per row.
        scope: token-granular scoping flags ``scope[i][j]``, same truthiness
            contract as :func:`golden_attention`, or None (= all allowed).
        base: RoPE base (law math-rope default 10000.0).
        query_positions: OPTIONAL per-row query override; entry ``r`` may be
            None (= default below), one global token index, or a LIST of
            them (several probes over the same chain share one key walk).
            For a SINGLE-row call the whole sequence may be given directly
            instead of one entry. Purely additive keyword.

    Returns:
        ``list[list[float]]`` — ONE output vector (length ``d_head``) per
        requested query, in row order. The queried token defaults to the
        sequence's LAST VALID token (the decode step).

    Raises:
        RuntimeError: on row/seq_lens mismatch, invalid seq_len, seq_len
            beyond chain capacity, a needed page-table entry without a block
            payload, stored tokens fewer than the sequence needs, mismatched
            head dims, an out-of-range query token, a fully masked query, or
            malformed bias/scope indexing.
    """
    try:
        n_rows = len(page_table_rows)
    except TypeError:
        raise RuntimeError(
            f"page_table_rows must be a sequence of rows, got {type(page_table_rows).__name__}"
        )
    try:
        n_lens = len(seq_lens)
    except TypeError:
        raise RuntimeError(f"seq_lens must be a sequence, got {type(seq_lens).__name__}")
    if n_rows != n_lens:
        raise RuntimeError(
            f"page_table_rows ({n_rows}) and seq_lens ({n_lens}) length mismatch"
        )
    if query_positions is None:
        qp_row: list = [None] * n_rows
    else:
        try:
            qp_spec = list(query_positions)
        except TypeError:
            raise RuntimeError(
                f"query_positions must be a sequence, got {type(query_positions).__name__}"
            )
        if len(qp_spec) == n_rows:
            qp_row = qp_spec  # entry r selects the query/queries of row r
        elif n_rows == 1:
            qp_row = [qp_spec]  # whole list = probes for the single row
        else:
            raise RuntimeError(
                f"query_positions ({len(qp_spec)}) must match page_table_rows "
                f"({n_rows}) entry-for-entry"
            )

    adapted: dict[int, tuple] = {}  # pid -> (toks, n_stored), adapted once
    outputs: list[list[float]] = []

    for r in range(n_rows):
        row = page_table_rows[r]
        try:
            width = len(row)
        except TypeError:
            raise RuntimeError(f"page-table row {r} must be a finite sequence")
        seq_len = seq_lens[r]
        if (
            isinstance(seq_len, bool)
            or not isinstance(seq_len, int)
            or seq_len < 1
        ):
            raise RuntimeError(f"seq_len must be an integer >= 1, got {seq_len!r}")
        needed = -(-seq_len // BLOCK_SIZE)  # ceil division
        if width < needed:
            raise RuntimeError(
                f"seq_len {seq_len} exceeds chain capacity of row {r} "
                f"({width} block(s) = {width * BLOCK_SIZE} tokens)"
            )
        phys_ids: list = []
        for entry in row:
            if entry == PAD_ID:
                continue
            phys_ids.append(entry)
            if len(phys_ids) == needed:
                break
        if len(phys_ids) < needed:
            raise RuntimeError(
                f"seq_len {seq_len} exceeds usable (non-padded) chain capacity "
                f"of row {r}"
            )

        resolved = []
        valid_counts = []
        for b_idx, pid in enumerate(phys_ids):
            cached = adapted.get(pid)
            if cached is None:
                try:
                    present = pid in blocks
                except TypeError:
                    present = False
                if not present:
                    raise RuntimeError(
                        f"page table entry {b_idx} -> physical block {pid!r} "
                        f"has no block payload"
                    )
                cached = _mt_block_tokens(blocks[pid], pid)
                adapted[pid] = cached
            resolved.append(cached[0])
            valid_counts.append(min(BLOCK_SIZE, seq_len - b_idx * BLOCK_SIZE))

        # ---- query selection (default: last valid token, the decode step) --
        # Each row's query_positions entry may be: None -> default, a single
        # global token index, or a LIST of them (several decode probes over
        # the same chain share one key walk).
        spec = qp_row[r]
        q_list: list[int]
        if spec is None:
            q_list = []
        elif isinstance(spec, bool):
            raise RuntimeError(f"invalid query token: {spec!r}")
        elif isinstance(spec, int):
            q_list = [spec]
        elif isinstance(spec, (list, tuple)):
            if not spec:
                raise RuntimeError(
                    f"query_positions entry for row {r} is empty"
                )
            for x in spec:
                if isinstance(x, bool) or not isinstance(x, int):
                    raise RuntimeError(f"invalid query token: {x!r}")
            q_list = list(spec)
        else:
            raise RuntimeError(
                f"invalid query_positions entry for row {r}: {spec!r}"
            )
        if not q_list:
            q_list = [seq_len - 1]
        for q_tok in q_list:
            if q_tok < 0 or q_tok >= seq_len:
                raise RuntimeError(
                    f"query token {q_tok!r} out of range for seq_len "
                    f"{seq_len} (row {r})"
                )

        # Walk the VALID keys/values once per row: (global position, vector).
        key_entries: list[tuple[int, list[float]]] = []
        value_entries: list[list[float]] = []
        for b_idx in range(needed):
            k_vecs = resolved[b_idx]["k"]
            v_vecs = resolved[b_idx]["v"]
            cnt = valid_counts[b_idx]
            if len(k_vecs) < cnt or len(v_vecs) < cnt:
                raise RuntimeError(
                    f"physical block {phys_ids[b_idx]!r} stores "
                    f"{min(len(k_vecs), len(v_vecs))} token(s); sequence "
                    f"needs {cnt}"
                )
            for slot in range(cnt):
                key_entries.append((b_idx * BLOCK_SIZE + slot, k_vecs[slot]))
                value_entries.append(v_vecs[slot])

        row_outputs: list[list[float]] = []
        for q_tok in q_list:
            q_block, q_slot = divmod(q_tok, BLOCK_SIZE)
            q_vecs = resolved[q_block]["q"]
            if q_slot >= len(q_vecs):
                raise RuntimeError(
                    f"physical block {phys_ids[q_block]!r} stores "
                    f"{len(q_vecs)} token(s); query slot {q_slot} missing"
                )
            q_vec = q_vecs[q_slot]
            d_head = _require_even_vector(q_vec, "query vector")
            scale = 1.0 / math.sqrt(float(d_head))
            q_rot = _rope_at(
                q_vec, q_tok, cos_table, sin_table, base, "query vector"
            )

            # ---- (3)-(6): rope/dot/bias/scope over ALL valid keys ----------
            scores: list[float] = []
            try:
                for k_pos, k_vec in key_entries:
                    if len(k_vec) != d_head:
                        raise RuntimeError(
                            f"key vector at global token {k_pos} has dim "
                            f"{len(k_vec)}, expected {d_head}"
                        )
                    k_rot = _rope_at(
                        k_vec, k_pos, cos_table, sin_table, base, "key vector"
                    )
                    s = sum(a * b for a, b in zip(q_rot, k_rot)) * scale
                    if bias is not None:
                        s += float(bias[q_tok][k_pos])
                    if scope is not None and not scope[q_tok][k_pos]:
                        s = float("-inf")
                    scores.append(s)
            except IndexError:
                raise RuntimeError(
                    f"bias/scope indexing failed for query token {q_tok}: "
                    f"matrices must cover [seq_len][seq_len] global indices"
                )
            if all(s == float("-inf") for s in scores):
                raise RuntimeError(
                    f"query token {q_tok} is fully masked: "
                    f"no key contributes weight"
                )

            # ---- (7) stable softmax over the sequence + weighted V sum -----
            m = max(scores)
            weights = [math.exp(s - m) for s in scores]
            total = sum(weights)

            out = [0.0] * d_head
            for w, v_vec in zip(weights, value_entries):
                if len(v_vec) != d_head:
                    raise RuntimeError(
                        f"value vector dim {len(v_vec)} != query dim {d_head}"
                    )
                if w == 0.0:
                    continue  # masked column contributes nothing
                for dd in range(d_head):
                    out[dd] += w * float(v_vec[dd])
            row_outputs.append([x / total for x in out])

        outputs.extend(row_outputs)

    return outputs
