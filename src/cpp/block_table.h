// Ouroboros v7.1 — Block-level memory management, production implementation.
//
// Array-backed linked list mirroring src/block_table.py semantics EXACTLY
// (law sys-blocks: the production table is C++). The pybind11 module
// `ouroboros_cpp` exposes this class for differential parity testing against
// the Python reference.
//
// INVARIANT (slot identity): a block id is forever equal to its storage
// index — payload_[i] / next_ptr_[i] always describe block i, across any
// number of alloc/free cycles. free_block() zeroes the slot payload and
// rejects double frees.

#ifndef OUROBOROS_BLOCK_TABLE_H_
#define OUROBOROS_BLOCK_TABLE_H_

#include <vector>

namespace ouroboros {

inline constexpr int kBlockSize = 64;  // Fixed 64-token physical block size
inline constexpr int kPadId = -1;      // Page-table "no physical block" marker

class BlockTable {
 public:
  explicit BlockTable(int max_blocks = 1024);

  int max_blocks() const { return max_blocks_; }
  int block_size() const { return kBlockSize; }
  int num_free() const { return static_cast<int>(free_list_.size()); }
  bool is_allocated(int idx) const;

  // Allocates the lowest free id first (mirrors the Python free-stack init).
  // Throws std::runtime_error when no free blocks are available.
  int allocate_block();

  // Returns the block to the free list (LIFO). Double-free throws and leaves
  // the table unchanged up to the point of the raise.
  void free_block(int idx);

  // Mutable payload of an allocated block (kBlockSize ints). Throws if the
  // slot is not allocated.
  int* get_block(int idx);
  const int* get_block(int idx) const;

  void set_cell(int idx, int offset, int value);
  int get_cell(int idx, int offset) const;

  int next_ptr_value(int idx) const;
  void link(int prev_idx, int next_idx);

  // All-or-nothing: validates before consuming any slot. Returns HEAD id.
  int allocate_chain(int n_blocks);

  // Block ids from head along next_ptr until the -1 marker. Validates head
  // eagerly at call time.
  std::vector<int> walk(int head) const;

  int chain_len(int head) const;

  // [EXPAND]: allocate ONE block, link tail -> new block, return new tail.
  // Refuses a non-tail block.
  int expand_chain(int tail);

  // Physical id at logical position logical_idx along the chain from head.
  int resolve_block(int head, int logical_idx) const;

  // Page-table row: physical ids along the chain (optionally right-padded
  // with kPadId), the exact layout the attention kernel consumes.
  std::vector<int> page_table_row(int head, int pad_to = -1) const;

 private:
  void require_allocated(int idx) const;

  int max_blocks_;
  std::vector<int> payload_;      // max_blocks_ * kBlockSize, zeroed slots
  std::vector<char> allocated_;   // 0/1 per slot
  std::vector<int> next_ptr_;     // -1 marks a chain tail / unallocated
  std::vector<int> free_list_;    // LIFO stack: back() is handed out next
};

}  // namespace ouroboros

#endif  // OUROBOROS_BLOCK_TABLE_H_
