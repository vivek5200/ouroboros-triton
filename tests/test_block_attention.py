"""Tests for the fused block-attention Triton kernel (paper §4.3).

Two tiers:

* **GPU tier** — kernel vs dense-reference correctness on a random small
  page table (2 sequences x 3 blocks). Each test calls
  ``pytest.importorskip`` so it SKIPS with a reason when triton/torch/CUDA
  are absent (the current host) instead of erroring.
* **Pure-python tier** — page-table address resolution and bias/mask shape
  validation helpers. These RUN on any host, no torch/triton required.
"""

import pytest

from src.block_table import (
    BLOCK_SIZE,
    PAD_ID,
    BlockTable,
    build_page_table,
    resolve_block,
)
from src.kernels import block_attention as ba


# ---------------------------------------------------------------------------
# Pure-python tier — runs everywhere (no torch / triton needed)
# ---------------------------------------------------------------------------


def test_module_imports_without_triton():
    """Importing the module must never crash without triton installed."""
    assert isinstance(ba.HAS_TRITON, bool)
    assert isinstance(ba.HAS_TORCH, bool)
    # The wrapper is always importable; only calling it demands the GPU stack.
    assert callable(ba.block_attention)
    assert callable(ba.validate_bias_shape)
    assert callable(ba.validate_mask_shape)


# -- page-table address resolution ------------------------------------------


def _small_table():
    bt = BlockTable(max_blocks=16)
    h1 = bt.allocate_chain(3)          # physical 0,1,2
    h2 = bt.allocate_chain(2)          # physical 3,4
    return bt, [h1, h2]


def test_resolve_block_row_returns_physical_id():
    bt, heads = _small_table()
    row = build_page_table(bt, heads, pad_to=4)[0]
    assert resolve_block(row, 0) == row[0]
    assert resolve_block(row, 2) == row[2]
    assert isinstance(resolve_block(row, 0), int)


def test_resolve_block_row_rejects_out_of_range_and_pad():
    bt, heads = _small_table()
    row = build_page_table(bt, heads, pad_to=4)[0]
    assert len(row) == 4 and row[3] == PAD_ID
    with pytest.raises(RuntimeError, match="padded"):
        resolve_block(row, 3)
    with pytest.raises(RuntimeError, match="out of range"):
        resolve_block(row, 4)
    with pytest.raises(RuntimeError, match="out of range"):
        resolve_block(row, -1)
    with pytest.raises(RuntimeError, match="invalid logical index"):
        resolve_block(row, True)  # bool is not a valid index


def test_resolve_block_rejects_raw_blocktable_instance():
    bt, heads = _small_table()
    with pytest.raises(RuntimeError, match="page-table row"):
        resolve_block(bt, 0)


def test_block_table_method_resolve_block_walks_chain():
    bt, heads = _small_table()
    head = heads[0]
    chain = list(bt.walk(head))
    for logical, phys in enumerate(chain):
        assert bt.resolve_block(head, logical) == phys
    # agrees with the free-function path over the same table state
    row = build_page_table(bt, heads)[0]
    assert all(
        bt.resolve_block(head, i) == resolve_block(row, i) for i in range(3)
    )
    with pytest.raises(RuntimeError, match="out of range"):
        bt.resolve_block(head, 3)


def test_page_table_row_padding_matches_kernel_layout():
    bt, heads = _small_table()
    rows = build_page_table(bt, heads, pad_to=5)
    assert len(rows) == 2
    assert rows[0] == [heads[0], heads[0] + 1, heads[0] + 2, PAD_ID, PAD_ID]
    assert rows[1] == [heads[1], heads[1] + 1, PAD_ID, PAD_ID, PAD_ID]
    with pytest.raises(RuntimeError, match="shorter than chain"):
        build_page_table(bt, heads[:1], pad_to=2)


def test_page_table_row_from_expand_chain_growth():
    """[EXPAND] flow: growing the chain extends the kernel-visible row."""
    bt, heads = _small_table()
    tail = heads[0] + 2
    new_tail = bt.expand_chain(tail)
    row = bt.page_table_row(heads[0])
    assert row[-1] == new_tail and len(row) == 4


# -- bias / mask shape validation -------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (64,),                       # per-column shared bias
        (128, 64),                   # shared across seqs + heads
        (2, 128, 64),                # per-head
        (1, 2, 128, 64),             # singleton batch
        (8, 2, 128, 64),             # fully materialised
    ],
)
def test_bias_shape_accepts_broadcastable_ranks(shape):
    ba.validate_bias_shape(shape, num_seqs=8, num_heads=2, q_len=128, kv_len=64)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 64),                     # wrong H
        (2, 127, 64),                # wrong Q
        (1, 2, 128, 65),             # wrong K
        (1, 1, 2, 128, 64),          # rank > 4
    ],
)
def test_bias_shape_rejects_incompatible(shape):
    with pytest.raises(RuntimeError):
        ba.validate_bias_shape(shape, num_seqs=8, num_heads=2, q_len=128, kv_len=64)


@pytest.mark.parametrize(
    "shape",
    [(7,), (2, 7), (1, 2, 7), (9, 2, 7)],
)
def test_mask_shape_accepts_broadcastable_ranks(shape):
    ba.validate_mask_shape(shape, num_seqs=9, num_q_blocks=2, num_kv_blocks=7)


@pytest.mark.parametrize(
    "shape",
    [(2, 8), (3, 2, 7), (1, 1, 2, 7)],
)
def test_mask_shape_rejects_incompatible(shape):
    with pytest.raises(RuntimeError):
        ba.validate_mask_shape(shape, num_seqs=9, num_q_blocks=2, num_kv_blocks=7)


def test_broadcast_strides_use_zero_for_broadcast_axes():
    assert ba._broadcast_strides((1, 2, 4, 4), (8, 2, 4, 4)) == (0, 16, 4, 1)
    # (H, Q, K) bias is shared across sequences -> S axis gets stride 0
    assert ba._broadcast_strides((2, 4, 4), (8, 2, 4, 4)) == (0, 16, 4, 1)
    # fully materialised rank-4 shape keeps every stride
    assert ba._broadcast_strides((8, 2, 4, 4), (8, 2, 4, 4)) == (32, 16, 4, 1)
    assert ba._broadcast_strides((2, 7), (9, 2, 7)) == (0, 7, 1)
    with pytest.raises(RuntimeError):
        ba._broadcast_strides((3, 4, 4), (8, 2, 4, 4))


# ---------------------------------------------------------------------------
# GPU tier — skipped with reasons until torch + triton + CUDA arrive
# ---------------------------------------------------------------------------


def _gpu_stack():
    """importorskip triton+torch, then demand CUDA. Returns the torch module."""
    pytest.importorskip("triton", reason="triton not installed on this host")
    torch = pytest.importorskip("torch", reason="torch not installed on this host")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available (kernel needs a GPU)")
    if not ba.HAS_TRITON:  # importorskip passed but module bound before install
        pytest.skip("src.kernels.block_attention imported without triton binding")
    return torch


def _build_random_table(torch, seed, seq_lens=(192, 150), num_heads=2, head_dim=32,
                        device="cuda", dtype=None):
    """2 sequences x 3 blocks of 64 tokens -> scattered caches via BlockTable."""
    dtype = torch.float32 if dtype is None else dtype
    g = torch.Generator(device="cpu").manual_seed(seed)
    bt = BlockTable(max_blocks=16)
    n_phys = sum((l + BLOCK_SIZE - 1) // BLOCK_SIZE for l in seq_lens)
    heads = [bt.allocate_chain((l + BLOCK_SIZE - 1) // BLOCK_SIZE) for l in seq_lens]
    max_b = max((l + BLOCK_SIZE - 1) // BLOCK_SIZE for l in seq_lens)
    page_table = build_page_table(bt, heads, pad_to=max_b)

    q_cache = torch.randn(n_phys, num_heads, BLOCK_SIZE, head_dim, generator=g).to(dtype=dtype, device=device)
    k_cache = torch.randn(n_phys, num_heads, BLOCK_SIZE, head_dim, generator=g).to(dtype=dtype, device=device)
    v_cache = torch.randn(n_phys, num_heads, BLOCK_SIZE, head_dim, generator=g).to(dtype=dtype, device=device)

    max_pos = max_b * BLOCK_SIZE
    half = head_dim // 2
    inv = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float64) * 2.0 / head_dim))
    ang = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv[None, :]
    cos = torch.cos(ang).float()
    sin = torch.sin(ang).float()

    pt = torch.tensor(page_table, dtype=torch.int32, device=device)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return bt, pt, q_cache, k_cache, v_cache, lens, cos.to(device), sin.to(device)


def _assert_close(got, want, atol, rtol, label):
    # torch-free comparison: this module must stay importable on CPU-only
    # machines, so the helper never references the module-global `torch`
    # (tests that need it bind a local `torch = _gpu_stack()`).
    g = got.detach().flatten().tolist()
    w = want.detach().flatten().tolist()
    max_diff = max(abs(a - b) for a, b in zip(g, w))
    ok = all(abs(a - b) <= atol + rtol * abs(b) for a, b in zip(g, w))
    assert ok, f"{label}: max abs diff {max_diff:.3e} exceeds atol={atol} rtol={rtol}"


def test_kernel_matches_dense_reference_with_rope_bias_mask():
    """Random small table (2 seqs x 3 blocks): full feature stack vs oracle."""
    torch = _gpu_stack()
    _, pt, q, k, v, lens, cos, sin = _build_random_table(torch, seed=1234)
    S, H, maxlen = 2, 2, max(lens).item()
    gen = torch.Generator().manual_seed(99)
    bias = torch.randn(1, H, maxlen, maxlen, generator=gen) * 0.25
    scope = torch.randint(0, 2, (S, 3, 3), generator=gen)
    scope |= torch.eye(3, dtype=torch.int64)  # every q-block keeps >= 1 key block
    mask = scope.to(torch.int8)

    out = ba.block_attention(pt, q, k, v, lens, cos, sin, bias=bias.cuda(), mask=mask.cuda())
    ref = ba.dense_reference(pt.cpu(), q.cpu(), k.cpu(), v.cpu(),
                             lens.tolist(), cos.cpu(), sin.cpu(), bias=bias, mask=mask)
    _assert_close(out.cpu(), ref, atol=2e-2, rtol=2e-2, label="rope+bias+mask")


def test_kernel_matches_dense_reference_no_bias_no_mask():
    torch = _gpu_stack()
    _, pt, q, k, v, lens, cos, sin = _build_random_table(torch, seed=77)
    out = ba.block_attention(pt, q, k, v, lens, cos, sin)
    ref = ba.dense_reference(pt.cpu(), q.cpu(), k.cpu(), v.cpu(),
                             lens.tolist(), cos.cpu(), sin.cpu())
    _assert_close(out.cpu(), ref, atol=2e-2, rtol=2e-2, label="plain rope attention")


def test_kernel_half_precision_path():
    torch = _gpu_stack()
    _, pt, q, k, v, lens, cos, sin = _build_random_table(torch, seed=5, dtype=torch.float16)
    out = ba.block_attention(pt, q, k, v, lens, cos, sin)
    ref = ba.dense_reference(pt.cpu(), q.cpu(), k.cpu(), v.cpu(),
                             lens.tolist(), cos.cpu(), sin.cpu())
    _assert_close(out.cpu(), ref, atol=5e-2, rtol=5e-2, label="fp16 attention")


def test_wrapper_validates_before_launch():
    """Shape/entry validation fires even before any GPU work is queued."""
    torch = _gpu_stack()
    _, pt, q, k, v, lens, cos, sin = _build_random_table(torch, seed=1)
    bad_pt = pt.clone()
    bad_pt[0, 0] = 999  # out of range for 6 physical blocks
    with pytest.raises(RuntimeError, match="out of range"):
        ba.block_attention(bad_pt, q, k, v, lens, cos, sin)
    with pytest.raises(RuntimeError, match="block dim"):
        ba.block_attention(pt, q[:, :, :32, :], k, v, lens, cos, sin)
