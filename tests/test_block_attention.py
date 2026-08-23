"""Tests for the fused block-attention Triton kernel (paper §4.3).

Three groups:

* **GPU tier** — kernel vs dense-reference correctness on a random small
  page table (2 sequences x 3 blocks). Each test calls
  ``pytest.importorskip`` so it SKIPS with a reason when triton/torch/CUDA
  are absent (the current host) instead of erroring.
* **Pure-python tier** — page-table address resolution and bias/mask shape
  validation helpers. These RUN on any host, no torch/triton required.
* **RoPE layout bridge** — the documented interleaved -> half-split channel
  permutation (:func:`src.kernels.reference_attention.perm` /
  ``apply_channel_perm`` / ``invert_channel_perm``): pure-python law tests
  run everywhere, and the kernel-vs-golden gap tests at ``d_head=8`` (dense
  oracle) and ``d_head=32`` (live Triton kernel) are GPU-gated. See the
  "RoPE layout" note in the reference module docstring.
"""

import math
import random

import pytest

from src.block_table import (
    BLOCK_SIZE,
    PAD_ID,
    BlockTable,
    build_page_table,
    resolve_block,
)
from src.kernels import block_attention as ba
from src.kernels import reference_attention as ra


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
# RoPE layout bridge — pure-python tier, runs everywhere (no torch/triton)
# ---------------------------------------------------------------------------


def test_perm_maps_interleaved_to_half_split():
    """perm(t) = (t%2)*half + t//2: formula, bijectivity, exact round-trips."""
    assert [ra.perm(t, 4) for t in range(8)] == [0, 4, 1, 5, 2, 6, 3, 7]
    assert [ra.perm(t, 1) for t in range(2)] == [0, 1]        # d=2: identity
    assert [ra.perm(t, 2) for t in range(4)] == [0, 2, 1, 3]  # d=4: swap middle
    for d in (2, 4, 6, 8, 10, 16, 32, 64):
        half = d // 2
        p = [ra.perm(t, half) for t in range(d)]
        assert sorted(p) == list(range(d))                    # bijection
        vec = list(range(d))
        assert ra.invert_channel_perm(ra.apply_channel_perm(vec, half), half) == vec
        assert ra.apply_channel_perm(ra.invert_channel_perm(vec, half), half) == vec
    with pytest.raises(RuntimeError):
        ra.apply_channel_perm([1.0, 2.0, 3.0, 4.0], 3)        # 2*half != len
    with pytest.raises(RuntimeError):
        ra.perm(8, 4)                                         # out of domain


def test_perm_involution_only_at_d_head_2_and_4():
    """Mathematical verification of the involution claim.

    perm is the classic out-shuffle on d = 2*half slots, so order(perm) is the
    multiplicative order of 2 mod (d-1); it is an involution iff (d-1) | 3,
    i.e. iff d_head in {2, 4}. At d_head=8 the order is 3.
    """
    for d in range(2, 66, 2):
        half = d // 2
        squared = [ra.perm(ra.perm(t, half), half) for t in range(d)]
        if d in (2, 4):
            assert squared == list(range(d)), f"d_head={d} must be involutive"
        else:
            assert squared != list(range(d)), f"d_head={d} must NOT be involutive"
    # d_head=8 concrete regime: cycles (1 4 2)(3 5 6), channels 0 and 7 fixed.
    assert [ra.perm(ra.perm(t, 4), 4) for t in range(8)] == [0, 2, 4, 6, 1, 3, 5, 7]
    v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    double_applied = ra.apply_channel_perm(ra.apply_channel_perm(v, 4), 4)
    assert double_applied != v      # NOT self-inverse at d_head=8
    assert ra.invert_channel_perm(ra.apply_channel_perm(v, 4), 4) == v
    # ... but the forward map IS its own inverse at the involutive widths.
    for half in (1, 2):
        w = [float(i + 1) for i in range(2 * half)]
        assert ra.apply_channel_perm(ra.apply_channel_perm(w, half), half) == w


def _half_split_rope(vec, position, base=10000.0):
    """Engine/kernel RoPE convention: pair t = (x[t], x[t+d/2]), theta_t = base**(-2t/d)."""
    d = len(vec)
    half = d // 2
    out = [0.0] * d
    for t in range(half):
        angle = float(position) * float(base) ** (-2.0 * t / d)
        c, s = math.cos(angle), math.sin(angle)
        x0, x1 = float(vec[t]), float(vec[t + half])
        out[t] = x0 * c - x1 * s
        out[t + half] = x0 * s + x1 * c
    return out


def test_channel_perm_bridges_golden_and_engine_rope_layouts():
    """The law behind the documented gap: perm(R_interleaved(x)) at position p
    == R_halfsplit(perm(x)) at position p — identical at d_head == 2."""
    rng = random.Random(20240711)
    for d in (2, 4, 8, 16, 32):
        half = d // 2
        for pos in (0, 1, 3, 17, 255):
            x = [rng.uniform(-1.0, 1.0) for _ in range(d)]
            lhs = ra.apply_channel_perm(ra.rope_rotate(x, pos), half)
            rhs = _half_split_rope(ra.apply_channel_perm(x, half), pos)
            for a, b in zip(lhs, rhs):
                assert abs(a - b) <= 1e-9 * (1.0 + abs(b))
    # d_head == 2: permutation is the identity and both layouts coincide.
    x = [0.5, -0.25]
    assert ra.apply_channel_perm(x, 1) == x
    assert ra.rope_rotate(x, 7) == _half_split_rope(x, 7)


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


# ---------------------------------------------------------------------------
# GPU tier — RoPE channel-permutation gap (kernel half-split vs golden
# interleaved-pair layouts). REGIME-RECORDING tests: each prints which regime
# held (un-permuted FAIL expected at d_head > 2; a PASS there would be a
# finding) and stays green as long as the permuted leg passes.
# ---------------------------------------------------------------------------


def _golden_rows(chain, blocks):
    """Golden (interleaved-layout) attention output row per chain token."""
    return [
        ra.golden_attention(list(chain), blocks, q_idx=i, bias=None, scope_mask=None)
        for i in range(len(chain))
    ]


def _materialize_paged_caches(torch, chain, blocks, back_pid, head_dim, device,
                              perm_channels):
    """Scatter the golden one-token-per-block payloads into engine caches.

    Layout bridge: the golden model keeps ONE token vector per physical block
    (token ``i`` lives on ``chain[i]``); the engine gathers whole 64-token
    blocks and truncates to ``seq_len`` rows. The equivalent engine layout
    therefore places token ``i`` at slot ``i % BLOCK_SIZE`` of a SINGLE
    backing block ``back_pid`` (page-table row ``[back_pid]``): after the
    gather + ``[:seq_len]`` truncate, gathered row ``i`` is token ``i`` at
    absolute RoPE position ``i`` — identical semantics on both sides.

    With ``perm_channels`` the payloads are relabeled interleaved->half-split
    on ingest — exactly how an engine cache must be laid out to be compared
    against the golden reference through :func:`ra.apply_channel_perm`.
    """
    n_phys = max(max(chain), back_pid) + 1
    shape = (n_phys, 1, BLOCK_SIZE, head_dim)
    q = torch.zeros(shape)
    k = torch.zeros(shape)
    v = torch.zeros(shape)
    half = head_dim // 2

    def _ch(vec):
        return ra.apply_channel_perm(vec, half) if perm_channels else list(vec)

    for i, pid in enumerate(chain):
        slot = i % BLOCK_SIZE
        q[back_pid, 0, slot, :] = torch.tensor(_ch(blocks[pid]["q"]))
        k[back_pid, 0, slot, :] = torch.tensor(_ch(blocks[pid]["k"]))
        v[back_pid, 0, slot, :] = torch.tensor(_ch(blocks[pid]["v"]))
    return q.to(device), k.to(device), v.to(device)


def _rope_tables(torch, max_pos, head_dim, base=10000.0):
    """Kernel-convention cos/sin tables [max_pos, D/2]: theta_t = base**(-2t/d)."""
    half = head_dim // 2
    inv = 1.0 / (float(base) ** (torch.arange(half, dtype=torch.float64) * 2.0 / head_dim))
    ang = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cos(ang).float(), torch.sin(ang).float()


def test_gpu_d8_rope_channel_permutation_gap():
    """[REGIME RECORDING] Documented RoPE-layout gap at d_head=8.

    The fused Triton kernel legally launches only at d_head >= 32 (wrapper
    guard + tl.dot needs K >= 16) -- asserted below -- so the kernel side of
    the d_head=8 comparison is ``dense_reference``, which carries the kernel's
    exact half-split RoPE/table convention (the existing GPU-tier tests pin
    kernel == dense_reference at d_head=32). Golden side: this module's
    interleaved-pair oracle.

    Legs:
      * RAW vs RAW   -> expected to FAIL at d_head=8 (that IS the documented
        gap); max_diff captured and the held regime printed.
      * kernel-side caches ingested with apply_channel_perm, golden output
        rows passed through apply_channel_perm -> MUST pass tolerance,
        proving the permutation is exactly the documented one.
    """
    torch = _gpu_stack()
    bt = BlockTable(max_blocks=16)
    T = 5
    head = bt.allocate_chain(T)      # golden chain: T one-token blocks
    back = bt.allocate_chain(1)      # engine backing block holding all T slots
    chain, blocks = ra.reference_from_block_table(bt, head, d_head=8)
    rows = _golden_rows(chain, blocks)

    # Record WHY no live launch happens at d_head=8 (wrapper guard).
    pt_row = build_page_table(bt, [back], pad_to=1)
    lens_cpu = torch.tensor([T], dtype=torch.int32)
    cos8, sin8 = _rope_tables(torch, BLOCK_SIZE, 8)
    q_raw, k_raw, v_raw = _materialize_paged_caches(
        torch, chain, blocks, back, 8, "cpu", perm_channels=False
    )
    with pytest.raises(RuntimeError, match="head dim"):
        ba.block_attention(torch.tensor(pt_row, dtype=torch.int32),
                           q_raw, k_raw, v_raw, lens_cpu, cos8, sin8)

    ref_dense_raw = ba.dense_reference(          # kernel convention, raw layout
        torch.tensor(pt_row, dtype=torch.int32), q_raw, k_raw, v_raw,
        [T], cos8, sin8,
    )[0, 0]

    tol = 1e-4  # fp32 dense math vs python-float64 golden: expect ~1e-6 noise

    def _diff(flat_a, flat_b):
        md = max(abs(a - b) for a, b in zip(flat_a, flat_b))
        ok = all(abs(a - b) <= tol for a, b in zip(flat_a, flat_b))
        return ok, md

    gold_flat = torch.tensor(rows).flatten().tolist()
    raw_flat = ref_dense_raw.flatten().tolist()
    unperm_ok, unperm_md = _diff(gold_flat, raw_flat)

    q_hs, k_hs, v_hs = _materialize_paged_caches(
        torch, chain, blocks, back, 8, "cpu", perm_channels=True
    )
    ref_dense_hs = ba.dense_reference(
        torch.tensor(pt_row, dtype=torch.int32), q_hs, k_hs, v_hs, [T], cos8, sin8
    )[0, 0]
    perm_flat = torch.tensor(
        [ra.apply_channel_perm(r, 4) for r in rows]
    ).flatten().tolist()
    perm_ok, perm_md = _diff(perm_flat, ref_dense_hs.flatten().tolist())

    # recorded-regime: un-permuted FAIL is the documented expectation at d=8;
    # an unexpected PASS is printed as a finding, not failed here ("assert
    # consistency either way") -- the permuted leg below is normative.
    regime = (
        "FINDING: UN-permuted already matched at d_head=8 (layouts coincided?!)"
        if unperm_ok
        else "documented gap: un-permuted kernel/reference outputs DISAGREE"
    )
    print(
        f"\n[rope-perm d_head=8] unperm max_diff={unperm_md:.3e} "
        f"({'PASS' if unperm_ok else 'FAIL'}); "
        f"permuted max_diff={perm_md:.3e} ({'PASS' if perm_ok else 'FAIL'}); "
        f"regime: {regime}"
    )

    assert perm_ok, (
        "apply_channel_perm'ed golden reference must match the "
        f"kernel-convention oracle at d_head=8 (max_diff={perm_md:.3e} > "
        f"tol={tol}) -- permutation is not the documented one"
    )
    if unperm_ok:
        print("[rope-perm d_head=8] FINDING recorded: un-permuted leg PASSED; "
              "investigate whether kernel/oracle RoPE layouts changed.")


def test_gpu_d32_live_kernel_matches_permuted_golden_reference():
    """[REGIME RECORDING] Closes the loop against the LIVE Triton kernel.

    At d_head=32 (smallest width the kernel accepts) the launched kernel's
    output equals apply_channel_perm(golden rows) within the fp32/tf32 GPU
    tolerance, while the raw-vs-raw comparison disagrees -- the documented
    permutation, verified end-to-end against real hardware execution.
    """
    torch = _gpu_stack()
    bt = BlockTable(max_blocks=16)
    T = 5
    head = bt.allocate_chain(T)      # golden chain: T one-token blocks
    back = bt.allocate_chain(1)      # engine backing block holding all T slots
    chain, blocks = ra.reference_from_block_table(bt, head, d_head=32)
    rows = _golden_rows(chain, blocks)

    pt = torch.tensor(build_page_table(bt, [back], pad_to=1),
                      dtype=torch.int32, device="cuda")
    lens = torch.tensor([T], dtype=torch.int32, device="cuda")
    cos, sin = _rope_tables(torch, BLOCK_SIZE, 32)

    atol, rtol = 2e-2, 2e-2  # mirrors the existing fp32 GPU-tier tolerances

    def _leg(perm_channels):
        q, k, v = _materialize_paged_caches(
            torch, chain, blocks, back, 32, "cuda", perm_channels=perm_channels
        )
        out = ba.block_attention(pt, q, k, v, lens, cos, sin)[0, 0].flatten().tolist()
        ref = rows if not perm_channels else [
            ra.apply_channel_perm(r, 16) for r in rows
        ]
        flat = torch.tensor(ref).flatten().tolist()
        ok = all(abs(a - b) <= atol + rtol * abs(b) for a, b in zip(out, flat))
        md = max(abs(a - b) for a, b in zip(out, flat))
        return ok, md

    unperm_ok, unperm_md = _leg(perm_channels=False)
    perm_ok, perm_md = _leg(perm_channels=True)
    regime = (
        "FINDING: UN-permuted live-kernel run matched the golden reference"
        if unperm_ok
        else "documented gap confirmed on hardware: raw layouts disagree"
    )
    print(
        f"\n[rope-perm d_head=32 live kernel] unperm max_diff={unperm_md:.3e} "
        f"({'PASS' if unperm_ok else 'FAIL'}); "
        f"permuted max_diff={perm_md:.3e} ({'PASS' if perm_ok else 'FAIL'}); "
        f"regime: {regime}"
    )

    assert perm_ok, (
        "permuted golden reference must match the live Triton kernel at "
        f"d_head=32 (max_diff={perm_md:.3e})"
    )
    if unperm_ok:
        print("[rope-perm d_head=32] FINDING recorded: un-permuted leg PASSED.")
