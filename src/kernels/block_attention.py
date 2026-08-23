"""Fused Triton pointer-chasing attention kernel (law sys-blocks, paper §4.3).

Block-level attention over a **page table of fixed 64-token physical blocks**
(``src.block_table.BLOCK_SIZE``). The kernel never materialises a dynamic
sequence length: every program resolves its logical coordinates
``(sequence, logical q-block / kv-block)`` to *scattered physical addresses*
by loading ids straight out of ``page_table`` on device — the classic paged
attention indirection — and does all tile work through ``tl.make_block_ptr``
loads/stores.

Per KV tile, entirely in SRAM/registers:

(a) **1-D RoPE rotation** of the q and k tiles using precomputed cos/sin
    tables passed as pointers (half-width tables ``[max_positions, D/2]``;
    rotation ``(x0, x1) -> (x0*c - x1*s, x0*s + x1*c)`` applied to the two
    feature halves). The score dot-product is split into two half-dots which
    sum to exactly the full-width dot — no concatenation needed.
(b) **Additive AST graph bias**: a bias matrix pointer is ADDED to the raw
    scores before the softmax (law math-rope: the graph bias is additive,
    NEVER a channel-split rotation).
(c) **Block-sparse scoping mask**: a per-(q-block, kv-block) scope mask
    loaded from a pointer; zero tiles are skipped outright (true sparsity),
    non-zero tiles attend.

Softmax is computed online (FlashAttention style: running max ``m``, running
sum ``l``, running accumulator ``acc``) and written through an output block
pointer with boundary checking so ragged tails are never stored.

IMPORT SAFETY: ``triton``/``torch`` are imported under ``try/except`` and
flipped into the ``HAS_TRITON`` / ``HAS_TORCH`` flags. The ``@triton.jit``
decorator runs at import time but the *kernel body* only compiles when the
kernel is first called (Triton JIT) — importing this module on a box with
no triton/torch never crashes; only calling :func:`block_attention` raises.

Memory layout contract (all device tensors, contiguous):

===================  =====================================  ====================
tensor               shape                                  dtype
===================  =====================================  ====================
``q/k/v_cache``      ``[n_phys_blocks, H, 64, D]``          fp16/bf16/fp32
``page_table``       ``[S, max_blocks_per_seq]``            int32/int64
``seq_lens``         ``[S]``                                int32/int64
``cos``/``sin``      ``[max_positions, D // 2]``            fp32
``bias``             broadcastable to ``[S, H, Q, K]``      fp16/bf16/fp32
``mask``             broadcastable to ``[S, QB, KB]``       int8/bool (0/≠0)
``out``              ``[S, H, max_len, D]``                 same as q_cache
===================  =====================================  ====================

Multi-head only (``H`` equal across q/k/v); GQA is out of scope here.
"""

import math
from typing import Sequence

try:  # pragma: no cover - exercised only on boxes with triton installed
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - the CI/dev baseline path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False

try:  # pragma: no cover - exercised only on boxes with torch installed
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - the CI/dev baseline path
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

from src.block_table import BLOCK_SIZE, PAD_ID

__all__ = [
    "HAS_TRITON",
    "HAS_TORCH",
    "block_attention",
    "dense_reference",
    "validate_bias_shape",
    "validate_mask_shape",
]


# ---------------------------------------------------------------------------
# Pure-python shape validation helpers (run WITHOUT torch/triton — tested now)
# ---------------------------------------------------------------------------


def validate_bias_shape(
    bias_shape: Sequence[int],
    *,
    num_heads: int,
    q_len: int,
    kv_len: int,
    num_seqs: int | None = None,
) -> None:
    """Validate an additive AST-graph-bias tensor shape against the score shape.

    The bias must be broadcastable to the score layout ``[S, H, Q, K]``
    (law math-rope: additive bias added pre-softmax, per head). Accepted
    ranks, right-aligned against ``(S, H, Q, K)`` with every concrete dim
    either matching the target or being 1 (broadcast):

    * ``(K,)`` / ``(1, K)`` — shared scalar-per-column bias
    * ``(Q, K)`` / ``(1, Q, K)`` — shared across sequences and heads
    * ``(H, Q, K)`` / ``(1, H, Q, K)`` — per-head, shared across sequences
    * ``(S, H, Q, K)`` — fully materialised

    Raises:
        RuntimeError: If the shape cannot broadcast to ``[S, H, Q, K]``.
    """
    shape = tuple(int(d) for d in bias_shape)
    targets = (num_seqs if num_seqs is not None else 1, num_heads, q_len, kv_len)
    if len(shape) > 4:
        raise RuntimeError(f"bias rank {len(shape)} > 4, cannot broadcast to [S, H, Q, K]")
    padded = (1,) * (4 - len(shape)) + shape
    for got, want in zip(padded, targets):
        if got != 1 and got != want:
            raise RuntimeError(
                f"bias dim {got} incompatible with expected {want} "
                f"(shape {shape} vs [S={targets[0]}, H={num_heads}, Q={q_len}, K={kv_len}])"
            )


def validate_mask_shape(
    mask_shape: Sequence[int],
    *,
    num_q_blocks: int,
    num_kv_blocks: int,
    num_seqs: int | None = None,
) -> None:
    """Validate a block-sparse scoping-mask shape.

    The scope mask lives at **block granularity**: one flag per
    (sequence, logical q-block, logical kv-block) triple, non-zero meaning
    "this query block may attend to this key block". Accepted ranks,
    right-aligned against ``(S, QB, KB)``:

    * ``(KB,)`` — per-kv-block scope shared by everything
    * ``(QB, KB)`` / ``(1, QB, KB)`` — shared across sequences
    * ``(S, QB, KB)`` — per-sequence scope

    Raises:
        RuntimeError: If the shape cannot broadcast to ``[S, QB, KB]``.
    """
    shape = tuple(int(d) for d in mask_shape)
    targets = (num_seqs if num_seqs is not None else 1, num_q_blocks, num_kv_blocks)
    if len(shape) > 3:
        raise RuntimeError(f"mask rank {len(shape)} > 3, cannot broadcast to [S, QB, KB]")
    if len(shape) == 1:
        padded = (1, 1, shape[0])
    else:
        padded = (1,) * (3 - len(shape)) + shape
    for got, want in zip(padded, targets):
        if got != 1 and got != want:
            raise RuntimeError(
                f"mask dim {got} incompatible with expected {want} "
                f"(shape {shape} vs [S={targets[0]}, QB={num_q_blocks}, KB={num_kv_blocks}])"
            )


def _broadcast_strides(shape: Sequence[int], target: Sequence[int]) -> tuple[int, ...]:
    """Row-major strides of ``shape`` right-aligned under ``target``; size-1 /
    absent axes get stride 0 so index arithmetic never walks out of bounds."""
    shape = tuple(int(d) for d in shape)
    if len(shape) > len(target):
        raise RuntimeError(f"rank {len(shape)} exceeds target rank {len(target)}")
    padded = (1,) * (len(target) - len(shape)) + shape
    for got, want in zip(padded, target):
        if got != 1 and got != want:
            raise RuntimeError(f"dim {got} incompatible with {want}")
    strides: list[int] = []
    running = 1
    for ax in range(len(target) - 1, -1, -1):
        strides.append(0 if padded[ax] == 1 else running)
        running *= padded[ax]
    return tuple(reversed(strides))


# ---------------------------------------------------------------------------
# Triton kernel — compiled ONLY on first call (JIT); definition is inert when
# triton is absent because the decorator never runs without the guard below.
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _block_attention_kernel(
        Q, K, V,                     # caches: [n_phys_blocks, H, BLOCK, D]
        PT,                          # page_table: [S, max_blocks_per_seq]
        LENS,                        # seq_lens: [S] (self-attention: q_len == kv_len)
        COS, SIN,                    # rope tables: [max_positions, D // 2]
        BIAS,                        # additive AST bias (pre-softmax), strided
        MASK,                        # block-sparse scope mask, strided
        OUT,                         # out: [S, H, max_len, D]
        sm_scale,
        qk_limit,                    # max_len: bias/score extent guard
        stride_qb, stride_qh, stride_qt,
        stride_kb, stride_kh, stride_kt,
        stride_vb, stride_vh, stride_vt,
        stride_pts, stride_ptb,
        stride_b0, stride_b1, stride_b2, stride_b3,
        stride_m0, stride_m1, stride_m2,
        stride_os, stride_oh, stride_ot,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HALF: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """One program == one (logical q-block, sequence, head).

        Pointer-chasing: the physical block id for every logical tile is
        LOADED from the page table on device, then used as the base offset
        of a make_block_ptr view into the scattered cache.
        """
        pid_m = tl.program_id(0)          # logical q-block index
        pid_sh = tl.program_id(1)         # seq * NUM_HEADS + head
        seq = pid_sh // NUM_HEADS
        h = pid_sh % NUM_HEADS

        q_len = tl.load(LENS + seq)       # self-attention: kv_len == q_len
        q_start = pid_m * BLOCK
        if q_start >= q_len:              # ragged grid: nothing to do here
            return

        offs_t = tl.arange(0, BLOCK)
        offs_d = tl.arange(0, HALF)

        # ---- logical -> physical resolution for THIS q tile (paper §4.3) ----
        q_phys = tl.load(PT + seq * stride_pts + pid_m * stride_ptb).to(tl.int64)

        # ---- RoPE tables at this tile's absolute positions -----------------
        q_pos = q_start + offs_t
        cs_off = q_pos[:, None] * HALF + offs_d[None, :]
        q_cos = tl.load(COS + cs_off)
        q_sin = tl.load(SIN + cs_off)

        # ---- scattered Q load via block ptr, split into feature halves -----
        q_base = Q + q_phys * stride_qb + h * stride_qh
        q0_bp = tl.make_block_ptr(
            base=q_base, shape=(BLOCK, HEAD_DIM), strides=(stride_qt, 1),
            offsets=(0, 0), block_shape=(BLOCK, HALF), order=(1, 0),
        )
        q1_bp = tl.make_block_ptr(
            base=q_base, shape=(BLOCK, HEAD_DIM), strides=(stride_qt, 1),
            offsets=(0, HALF), block_shape=(BLOCK, HALF), order=(1, 0),
        )
        q0 = tl.load(q0_bp).to(tl.float32)
        q1 = tl.load(q1_bp).to(tl.float32)

        # ---- (a) 1-D RoPE rotation of q, in SRAM ---------------------------
        q0r = q0 * q_cos - q1 * q_sin
        q1r = q0 * q_sin + q1 * q_cos

        m_i = tl.full([BLOCK], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK], tl.float32)
        acc = tl.zeros([BLOCK, HEAD_DIM], tl.float32)

        kv_len = q_len
        hi = tl.cdiv(kv_len, BLOCK)
        for n in range(0, hi):
            # ---- (c) block-sparse scoping: skip disallowed tiles outright --
            if HAS_MASK:
                scope = tl.load(
                    MASK + seq * stride_m0 + pid_m * stride_m1 + n * stride_m2
                )
                if scope == 0:
                    continue

            kv_phys = tl.load(PT + seq * stride_pts + n * stride_ptb).to(tl.int64)
            kv_pos = n * BLOCK + offs_t
            ks_off = kv_pos[:, None] * HALF + offs_d[None, :]
            k_cos = tl.load(COS + ks_off)
            k_sin = tl.load(SIN + ks_off)

            k_base = K + kv_phys * stride_kb + h * stride_kh
            k0_bp = tl.make_block_ptr(
                base=k_base, shape=(BLOCK, HEAD_DIM), strides=(stride_kt, 1),
                offsets=(0, 0), block_shape=(BLOCK, HALF), order=(1, 0),
            )
            k1_bp = tl.make_block_ptr(
                base=k_base, shape=(BLOCK, HEAD_DIM), strides=(stride_kt, 1),
                offsets=(0, HALF), block_shape=(BLOCK, HALF), order=(1, 0),
            )
            k0 = tl.load(k0_bp).to(tl.float32)
            k1 = tl.load(k1_bp).to(tl.float32)

            # ---- (a) 1-D RoPE rotation of k, in SRAM -----------------------
            k0r = k0 * k_cos - k1 * k_sin
            k1r = k0 * k_sin + k1 * k_cos

            # Two half-dots sum to exactly the full-width score dot product.
            qk = tl.dot(q0r, tl.trans(k0r)) + tl.dot(q1r, tl.trans(k1r))
            qk = qk * sm_scale

            # ---- (b) ADDITIVE AST graph bias, pre-softmax (law math-rope) --
            if HAS_BIAS:
                offs_n = kv_pos
                bias_tile = tl.load(
                    BIAS
                    + seq * stride_b0
                    + h * stride_b1
                    + (q_start + offs_t)[:, None] * stride_b2
                    + offs_n[None, :] * stride_b3,
                    mask=((q_start + offs_t)[:, None] < qk_limit)
                    & (offs_n[None, :] < qk_limit),
                    other=0.0,
                )
                qk = qk + bias_tile.to(tl.float32)

            # Ragged tail: columns past kv_len never contribute.
            qk = tl.where(kv_pos[None, :] < kv_len, qk, float("-inf"))

            # ---- online softmax (FlashAttention) ---------------------------
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(qk - m_ij[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)

            v_base = V + kv_phys * stride_vb + h * stride_vh
            v_bp = tl.make_block_ptr(
                base=v_base, shape=(BLOCK, HEAD_DIM), strides=(stride_vt, 1),
                offsets=(0, 0), block_shape=(BLOCK, HEAD_DIM), order=(1, 0),
            )
            v = tl.load(v_bp).to(tl.float32)
            acc = acc * alpha[:, None]
            acc = tl.dot(p, v, acc)
            m_i = m_ij

        acc = acc / l_i[:, None]

        # ---- store through an out block ptr; ragged rows never land --------
        o_base = OUT + seq * stride_os + h * stride_oh
        o_bp = tl.make_block_ptr(
            base=o_base, shape=(q_len, HEAD_DIM), strides=(stride_ot, 1),
            offsets=(q_start, 0), block_shape=(BLOCK, HEAD_DIM), order=(1, 0),
        )
        tl.store(o_bp, acc.to(OUT.dtype.element_ty), boundary_check=(0,))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def _require_gpu_stack() -> None:
    if not HAS_TRITON:
        raise RuntimeError(
            "block_attention requires triton (not installed on this host); "
            "the pure-python helpers and tests run without it"
        )
    if not HAS_TORCH:
        raise RuntimeError("block_attention requires torch (not installed on this host)")
    if not torch.cuda.is_available():
        raise RuntimeError("block_attention requires a CUDA device")


def _as_int32(t, name: str):
    if t.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"{name} must be int32/int64, got {t.dtype}")
    return t.to(torch.int32, copy=False).contiguous()


def block_attention(
    page_table,
    q_cache,
    k_cache,
    v_cache,
    seq_lens,
    cos,
    sin,
    bias=None,
    mask=None,
    softmax_scale: float | None = None,
):
    """Launch the fused block-level attention kernel (paper §4.3, law sys-blocks).

    Args mapping to the paper / laws:

    * ``page_table`` — **§4.3 block table**: ``[S, max_blocks_per_seq]`` int
      tensor; entry ``(s, n)`` is the physical 64-token block holding logical
      block ``n`` of sequence ``s``, or :data:`src.block_table.PAD_ID` for an
      unused slot. Built by :func:`src.block_table.build_page_table`. This is
      the only link between the C++/Python block manager and the GPU.
    * ``q_cache, k_cache, v_cache`` — **§4.3 scattered KV cache**:
      ``[n_phys_blocks, H, 64, D]`` each; rows are addressed *only* through
      physical ids resolved from ``page_table``.
    * ``seq_lens`` — per-sequence token counts ``[S]``; self-attention, so the
      query extent equals the key extent. Ragged tails are masked in-kernel.
    * ``cos, sin`` — **law math-rope**: precomputed RoPE tables
      ``[max_positions, D // 2]`` fp32 pointers applied as a 1-D rotation to
      q and k inside SRAM. Positions are absolute token offsets.
    * ``bias`` — **law math-rope / §4.3 AST graph bias**: ADDED to the raw
      scores pre-softmax (never a channel-split rotation). Broadcastable to
      ``[S, H, Q, K]`` — see :func:`validate_bias_shape`.
    * ``mask`` — **law sys-blocks block-sparse scope**: block-granularity
      ``[S?, QB, KB]`` flags, non-zero = attend. Zero tiles are skipped, so
      sparsity translates into real work avoided. See
      :func:`validate_mask_shape`.
    * ``softmax_scale`` — score scale; defaults to ``1/sqrt(D)``.

    Returns:
        ``[S, H, max_len, D]`` output tensor (rows past ``seq_lens[s]`` are zero).
    """
    _require_gpu_stack()
    S, max_b = page_table.shape
    n_phys, H, blk, D = q_cache.shape
    if blk != BLOCK_SIZE:
        raise RuntimeError(f"q_cache block dim must be {BLOCK_SIZE}, got {blk}")
    if k_cache.shape != q_cache.shape or v_cache.shape != q_cache.shape:
        raise RuntimeError("q/k/v caches must share the [n_phys, H, 64, D] layout")
    if D < 32 or D % 2 != 0:
        raise RuntimeError(f"head dim must be even and >= 32 for half-dot RoPE, got {D}")
    if tuple(seq_lens.shape) != (S,):
        raise RuntimeError(f"seq_lens must have shape [{S}], got {tuple(seq_lens.shape)}")

    # ---- validate page-table contents against the caches (host side) ------
    pt_list = _as_int32(page_table.contiguous(), "page_table").tolist()
    lens = _as_int32(seq_lens.contiguous()).tolist()
    max_len = 0
    for s_i, row in enumerate(pt_list):
        need = (lens[s_i] + BLOCK_SIZE - 1) // BLOCK_SIZE
        if need > max_b:
            raise RuntimeError(f"sequence {s_i} needs {need} blocks > table width {max_b}")
        for n_i, phys in enumerate(row):
            if n_i >= need:
                if phys != PAD_ID:
                    raise RuntimeError(
                        f"page_table[{s_i}][{n_i}] must be PAD_ID({PAD_ID}) beyond "
                        f"seq_len {lens[s_i]}, got {phys}"
                    )
            elif not (0 <= phys < n_phys):
                raise RuntimeError(
                    f"page_table[{s_i}][{n_i}]={phys} out of range for "
                    f"{n_phys} physical blocks"
                )
        max_len = max(max_len, lens[s_i])
    if max_len == 0:
        raise RuntimeError("all sequences are empty")
    if cos.shape[0] < max_b * BLOCK_SIZE or sin.shape[0] < max_b * BLOCK_SIZE:
        raise RuntimeError(
            f"rope tables must cover {max_b * BLOCK_SIZE} positions, got {cos.shape[0]}"
        )

    # ---- optional bias / mask normalization --------------------------------
    qb_max = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if bias is not None:
        validate_bias_shape(
            tuple(bias.shape), num_seqs=S, num_heads=H, q_len=max_len, kv_len=max_len
        )
        bias = bias.contiguous()
        b_strides = _broadcast_strides(tuple(bias.shape), (S, H, max_len, max_len))
    else:
        b_strides = (0, 0, 0, 0)
    if mask is not None:
        validate_mask_shape(
            tuple(mask.shape), num_seqs=S, num_q_blocks=qb_max, num_kv_blocks=qb_max
        )
        mask = mask.to(torch.int8).contiguous()
        m_strides = _broadcast_strides(tuple(mask.shape), (S, qb_max, qb_max))
    else:
        m_strides = (0, 0, 0)

    pt_dev = _as_int32(page_table, "page_table")
    lens_dev = _as_int32(seq_lens, "seq_lens")
    cos_c, sin_c = cos.contiguous(), sin.contiguous()
    q_c, k_c, v_c = q_cache.contiguous(), k_cache.contiguous(), v_cache.contiguous()
    # zeros, NOT empty: the grid is rounded up to whole blocks, so tail rows
    # past seq_len are owned by no program and must read back as defined zeros.
    out = torch.zeros((S, H, max_len, D), dtype=q_c.dtype, device=q_c.device)

    grid = (triton.cdiv(max_len, BLOCK_SIZE), S * H)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    sb, sh_, st = H * BLOCK_SIZE * D, BLOCK_SIZE * D, D

    _block_attention_kernel[grid](
        q_c, k_c, v_c,
        pt_dev,
        lens_dev,
        cos_c, sin_c,
        bias if bias is not None else q_c,   # dummy ptr when unused
        mask if mask is not None else q_c,   # dummy ptr when unused
        out,
        scale,
        float(max_len),
        sb, sh_, st,
        sb, sh_, st,
        sb, sh_, st,
        max_b, 1,
        *b_strides,
        *m_strides,
        H * max_len * D, max_len * D, D,
        NUM_HEADS=H,
        HEAD_DIM=D,
        HALF=D // 2,
        BLOCK=BLOCK_SIZE,
        HAS_BIAS=bias is not None,
        HAS_MASK=mask is not None,
    )
    return out


# ---------------------------------------------------------------------------
# Dense reference oracle (torch-only; test/diagnostics use)
# ---------------------------------------------------------------------------
# NOTE: law sys-blocks forbids dynamic reshapes on the *engine* path. This
# oracle deliberately ignores that constraint — it is the straightforward
# dense computation the kernel must match against.


def dense_reference(page_table, q_cache, k_cache, v_cache, seq_lens, cos, sin,
                    bias=None, mask=None, softmax_scale=None):
    """Dense attention over the same paged inputs; returns ``[S, H, maxlen, D]``.

    Uses the SAME page-table resolution rules (:data:`PAD_ID` handling) via
    :func:`src.block_table.resolve_block`, so a bug in address resolution
    cannot cancel out between kernel and oracle.
    """
    if not HAS_TORCH:
        raise RuntimeError("dense_reference requires torch")
    from src.block_table import resolve_block as _resolve

    S, max_b = page_table.shape
    _, H, blk, D = q_cache.shape
    hf = D // 2
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    lens = [int(x) for x in seq_lens]
    max_len = max(lens)
    pt_rows = [[int(x) for x in row] for row in page_table]
    qb_count = (max_len + blk - 1) // blk
    out = torch.zeros((S, H, max_len, D), dtype=torch.float32)

    def rope(x, pos):
        c = cos[pos][:, None, :]          # [T, 1, D/2]
        s = sin[pos][:, None, :]
        x0, x1 = x[..., :hf], x[..., hf:]
        return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)

    for s_i in range(S):
        T = lens[s_i]
        if T == 0:
            continue
        nb = (T + blk - 1) // blk
        ids = [_resolve(pt_rows[s_i], i) for i in range(nb)]
        # Gather the scattered physical blocks into a dense [H, T, D] tile.
        q = torch.cat([q_cache[p] for p in ids], dim=1)[:, :T, :].float()
        k = torch.cat([k_cache[p] for p in ids], dim=1)[:, :T, :].float()
        v = torch.cat([v_cache[p] for p in ids], dim=1)[:, :T, :].float()
        pos = torch.arange(T)
        q, k = rope(q, pos), rope(k, pos)
        scores = torch.einsum("htd,hsd->hts", q, k) * scale   # [H, T, T]
        if bias is not None:
            b4 = bias
            while b4.dim() < 4:
                b4 = b4.unsqueeze(0)
            b = b4.expand(S, H, max_len, max_len).float()[s_i][:, :T, :T]
            scores = scores + b
        if mask is not None:
            m3 = mask
            while m3.dim() < 3:
                m3 = m3.unsqueeze(0)
            sc = m3.expand(S, qb_count, qb_count).float()[s_i]       # [QB, KB]
            tok_q = torch.repeat_interleave(sc, blk, dim=0)[:T]      # [T, KB]
            tok = torch.repeat_interleave(tok_q, blk, dim=1)[:, :T]  # [T, T]
            scores = scores.masked_fill(tok == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out[s_i, :, :T, :] = torch.einsum("hts,hsd->htd", attn, v)
    return out
