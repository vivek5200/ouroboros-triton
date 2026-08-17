"""Module 2: Block-level memory management.

CONSTRAINT: Dynamic PyTorch tensor reshapes are FORBIDDEN.
All sequence expansion uses fixed 64-token physical blocks managed
via an array-backed linked list (block table).
"""

BLOCK_SIZE = 64  # Fixed 64-token physical block size


class BlockTable:
    """Array-backed linked list for VRAM block management.

    Each block holds exactly BLOCK_SIZE tokens. Blocks are linked
    via next-pointers stored in a flat array for cache coherency.

    TODO: Implement full block allocation, deallocation, and
    pointer-chasing attention support.
    """

    def __init__(self, max_blocks: int = 1024):
        self.max_blocks = max_blocks
        self.block_size = BLOCK_SIZE
        self.blocks: list[list[int]] = []
        self.next_ptr: list[int] = []  # Linked list next pointers
        self.free_list: list[int] = list(range(max_blocks))

    def allocate_block(self) -> int:
        """Allocate a free block and return its index.

        Returns:
            Block index.

        Raises:
            RuntimeError: If no free blocks available.
        """
        if not self.free_list:
            raise RuntimeError("No free blocks available")
        block_idx = self.free_list.pop(0)
        self.blocks.append([0] * self.block_size)
        self.next_ptr.append(-1)  # -1 = no next block
        return block_idx

    def free_block(self, block_idx: int) -> None:
        """Return a block to the free list."""
        self.free_list.append(block_idx)

    @property
    def num_free(self) -> int:
        return len(self.free_list)
