"""Module 2: Block-level memory management.

CONSTRAINT: Dynamic PyTorch tensor reshapes are FORBIDDEN.
All sequence expansion uses fixed 64-token physical blocks managed
via an array-backed linked list (block table).

INVARIANT (slot identity): a block id is forever equal to its storage
index — ``blocks[i]`` / ``next_ptr[i]`` always describe block ``i``, across
any number of alloc/free cycles. ``free_block`` clears the slot payload and
rejects double frees.
"""

BLOCK_SIZE = 64  # Fixed 64-token physical block size
PAD_ID = -1  # Page-table slot marker: "no physical block at this logical slot"


def resolve_block(table, logical_idx: int) -> int:
    """Resolve a logical block position to its physical block id (page-table lookup).

    This is the pure-Python twin of the address-translation step the Triton
    kernel performs in hardware (paper §4.3: logical coords → scattered
    physical addresses via the page table).

    Args:
        table: A 1-D page-table row — any sequence of physical block ids,
            one per logical slot, with ``PAD_ID`` (-1) marking an unused
            slot (e.g. produced by ``BlockTable.page_table_row`` /
            ``build_page_table``).
        logical_idx: Logical block position within the sequence (0-based).

    Returns:
        The physical block id stored at that logical slot.

    Raises:
        RuntimeError: If ``table`` is not a 1-D row, ``logical_idx`` is out
            of range, or the slot is padded (no backing physical block).
    """
    if isinstance(table, BlockTable):
        raise RuntimeError(
            "resolve_block expects a 1-D page-table row; use "
            "BlockTable.resolve_block(head, logical_idx) for chain lookup"
        )
    try:
        n_slots = len(table)
    except TypeError:
        raise RuntimeError(f"page table row must be a finite sequence, got {type(table).__name__}")
    if isinstance(logical_idx, bool) or not isinstance(logical_idx, int):
        raise RuntimeError(f"invalid logical index: {logical_idx!r}")
    if logical_idx < 0 or logical_idx >= n_slots:
        raise RuntimeError(
            f"logical block {logical_idx} out of range for page table of {n_slots} slots"
        )
    phys = table[logical_idx]
    if phys == PAD_ID:
        raise RuntimeError(f"logical block {logical_idx} is padded (no physical block)")
    return phys


class BlockTable:
    """Array-backed linked list for VRAM block management (fixed slots).

    Storage is pre-allocated as ``max_blocks`` slots so a block's id equals
    its index for the table's entire lifetime — the desync failure mode of
    append-on-allocate designs cannot occur here.
    """

    def __init__(self, max_blocks: int = 1024):
        self.max_blocks = max_blocks
        self.block_size = BLOCK_SIZE
        # Slot arrays: index == block id, forever.
        self.blocks: list[list[int] | None] = [None] * max_blocks
        self.next_ptr: list[int] = [-1] * max_blocks
        # Free stack, reversed so allocate_block() hands out id 0 first.
        self.free_list: list[int] = list(range(max_blocks))[::-1]

    def allocate_block(self) -> int:
        """Allocate a free block and return its id (== storage index).

        Raises:
            RuntimeError: If no free blocks available.
        """
        if not self.free_list:
            raise RuntimeError("No free blocks available")
        idx = self.free_list.pop()
        self.blocks[idx] = [0] * self.block_size
        self.next_ptr[idx] = -1
        return idx

    def free_block(self, idx: int) -> None:
        """Return an allocated block to the free list (double-free raises)."""
        self._require_allocated(idx)
        self.blocks[idx] = None
        self.next_ptr[idx] = -1
        self.free_list.append(idx)

    def get_block(self, idx: int) -> list[int]:
        """Payload of an allocated block."""
        self._require_allocated(idx)
        return self.blocks[idx]  # type: ignore[return-value]

    def link(self, prev_idx: int, next_idx: int) -> None:
        """Chain two allocated blocks (array-backed linked-list pointer)."""
        self._require_allocated(prev_idx)
        self._require_allocated(next_idx)
        self.next_ptr[prev_idx] = next_idx

    # ------------------------------------------------------------------
    # Sequence operations ([EXPAND]-aware primitives for the kernel
    # dispatch loop). Pure bookkeeping: stdlib only, no tensor reshapes.
    # ------------------------------------------------------------------

    def allocate_chain(self, n_blocks: int) -> int:
        """Allocate ``n_blocks`` blocks linked head→tail; return the HEAD id.

        All-or-nothing: the availability check happens before any slot is
        consumed, so a failed call leaves the table byte-for-byte unchanged.
        """
        if isinstance(n_blocks, bool) or not isinstance(n_blocks, int):
            raise RuntimeError(f"invalid block count: {n_blocks!r}")
        if n_blocks <= 0:
            raise RuntimeError(f"chain length must be >= 1, got {n_blocks}")
        if len(self.free_list) < n_blocks:
            raise RuntimeError(
                f"insufficient free blocks: requested {n_blocks}, "
                f"{len(self.free_list)} available"
            )
        ids = [self.allocate_block() for _ in range(n_blocks)]
        for prev, nxt in zip(ids, ids[1:]):
            self.next_ptr[prev] = nxt
        return ids[0]

    def walk(self, head: int):
        """Yield block ids from ``head`` along next_ptr until the -1 marker.

        Validates ``head`` eagerly at call time (a plain generator would
        defer the RuntimeError past the dispatch loop's guard).
        """
        self._require_allocated(head)
        return self._walk_from(head)

    def _walk_from(self, head: int):
        idx = head
        while idx != -1:
            yield idx
            idx = self.next_ptr[idx]

    def chain_len(self, head: int) -> int:
        """Number of blocks in the chain starting at ``head``."""
        self._require_allocated(head)
        return sum(1 for _ in self.walk(head))

    def expand_chain(self, tail: int) -> int:
        """[EXPAND]: allocate ONE block, link ``tail`` → new block.

        Returns the NEW tail id. Refuses a non-tail block so the dispatch
        loop cannot silently orphan a successor mid-chain.
        """
        self._require_allocated(tail)
        if self.next_ptr[tail] != -1:
            raise RuntimeError(f"block {tail} is not a chain tail")
        if not self.free_list:
            raise RuntimeError("No free blocks available")  # state unchanged
        new_idx = self.allocate_block()
        self.next_ptr[tail] = new_idx
        return new_idx

    def resolve_block(self, head: int, logical_idx: int) -> int:
        """Physical block id at logical position ``logical_idx`` of the chain.

        Kernel-side twin: the Triton kernel does this lookup per 64-token
        tile by loading the id straight out of the page table (paper §4.3).

        Raises:
            RuntimeError: If ``head`` is not an allocated block or
                ``logical_idx`` is out of range for the chain.
        """
        row = self.page_table_row(head)
        if isinstance(logical_idx, bool) or not isinstance(logical_idx, int):
            raise RuntimeError(f"invalid logical index: {logical_idx!r}")
        if logical_idx < 0 or logical_idx >= len(row):
            raise RuntimeError(
                f"logical block {logical_idx} out of range for chain of "
                f"{len(row)} block(s) starting at head {head}"
            )
        return row[logical_idx]

    def page_table_row(self, head: int, pad_to: int | None = None) -> list[int]:
        """Build a 1-D page-table row: physical ids along the chain from ``head``.

        ``pad_to`` right-pads with ``PAD_ID`` so every sequence's row has the
        same width — exactly the [num_seqs, max_blocks_per_seq] layout the
        kernel consumes.
        """
        row = list(self.walk(head))
        if pad_to is not None:
            if pad_to < len(row):
                raise RuntimeError(f"pad_to={pad_to} shorter than chain ({len(row)} blocks)")
            row.extend([PAD_ID] * (pad_to - len(row)))
        return row

    def _require_allocated(self, idx: int) -> None:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"invalid block id: {idx!r}")
        if idx < 0 or idx >= self.max_blocks or self.blocks[idx] is None:
            raise RuntimeError(f"block {idx} is not allocated")

    @property
    def num_free(self) -> int:
        return len(self.free_list)


def build_page_table(
    table: "BlockTable", heads, pad_to: int | None = None
) -> list[list[int]]:
    """Stack one page-table row per chain head → [num_seqs, max_blocks_per_seq].

    This is the exact tensor the block-attention kernel indexes on device:
    ``page_table[s][n]`` is the physical block holding logical block ``n``
    of sequence ``s`` (paper §4.3). Rows are padded to a common width with
    ``PAD_ID`` when ``pad_to`` is given.
    """
    return [table.page_table_row(head, pad_to=pad_to) for head in heads]
