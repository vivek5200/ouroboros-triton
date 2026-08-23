#include "block_table.h"

#include <stdexcept>
#include <string>

namespace ouroboros {

namespace {

[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error(msg);  // pybind11 maps this to Python RuntimeError
}

}  // namespace

BlockTable::BlockTable(int max_blocks)
    : max_blocks_(max_blocks),
      payload_(static_cast<size_t>(max_blocks) * kBlockSize, 0),
      allocated_(static_cast<size_t>(max_blocks), 0),
      next_ptr_(static_cast<size_t>(max_blocks), -1) {
  if (max_blocks <= 0) {
    fail("max_blocks must be >= 1");
  }
  // Free stack initialised REVERSED so allocate_block() hands out id 0
  // first — identical allocation order to the Python reference.
  free_list_.reserve(static_cast<size_t>(max_blocks));
  for (int i = max_blocks - 1; i >= 0; --i) {
    free_list_.push_back(i);
  }
}

bool BlockTable::is_allocated(int idx) const {
  return idx >= 0 && idx < max_blocks_ && allocated_[static_cast<size_t>(idx)] != 0;
}

void BlockTable::require_allocated(int idx) const {
  if (idx < 0 || idx >= max_blocks_ ||
      allocated_[static_cast<size_t>(idx)] == 0) {
    fail("block " + std::to_string(idx) + " is not allocated");
  }
}

int BlockTable::allocate_block() {
  if (free_list_.empty()) {
    fail("No free blocks available");
  }
  const int idx = free_list_.back();
  free_list_.pop_back();
  allocated_[static_cast<size_t>(idx)] = 1;
  next_ptr_[static_cast<size_t>(idx)] = -1;
  std::fill_n(payload_.begin() + static_cast<ptrdiff_t>(idx) * kBlockSize,
              kBlockSize, 0);
  return idx;
}

void BlockTable::free_block(int idx) {
  require_allocated(idx);  // double-free raises here
  allocated_[static_cast<size_t>(idx)] = 0;
  next_ptr_[static_cast<size_t>(idx)] = -1;
  std::fill_n(payload_.begin() + static_cast<ptrdiff_t>(idx) * kBlockSize,
              kBlockSize, 0);
  free_list_.push_back(idx);
}

int* BlockTable::get_block(int idx) {
  require_allocated(idx);
  return payload_.data() + static_cast<ptrdiff_t>(idx) * kBlockSize;
}

const int* BlockTable::get_block(int idx) const {
  require_allocated(idx);
  return payload_.data() + static_cast<ptrdiff_t>(idx) * kBlockSize;
}

void BlockTable::set_cell(int idx, int offset, int value) {
  get_block(idx)[offset] = value;
}

int BlockTable::get_cell(int idx, int offset) const {
  return get_block(idx)[offset];
}

int BlockTable::next_ptr_value(int idx) const {
  require_allocated(idx);
  return next_ptr_[static_cast<size_t>(idx)];
}

void BlockTable::link(int prev_idx, int next_idx) {
  require_allocated(prev_idx);
  require_allocated(next_idx);
  next_ptr_[static_cast<size_t>(prev_idx)] = next_idx;
}

int BlockTable::allocate_chain(int n_blocks) {
  // All-or-nothing: validate before consuming any slot so a failed call
  // leaves the table byte-for-byte unchanged.
  if (n_blocks <= 0) {
    fail("chain length must be >= 1, got " + std::to_string(n_blocks));
  }
  if (static_cast<int>(free_list_.size()) < n_blocks) {
    fail("insufficient free blocks: requested " + std::to_string(n_blocks) +
         ", " + std::to_string(free_list_.size()) + " available");
  }
  int head = -1;
  int prev = -1;
  for (int i = 0; i < n_blocks; ++i) {
    const int cur = allocate_block();
    if (i == 0) {
      head = cur;
    } else {
      next_ptr_[static_cast<size_t>(prev)] = cur;
    }
    prev = cur;
  }
  return head;
}

std::vector<int> BlockTable::walk(int head) const {
  require_allocated(head);  // eager validation, mirrors the Python wrapper
  std::vector<int> ids;
  int idx = head;
  while (idx != -1) {
    ids.push_back(idx);
    idx = next_ptr_[static_cast<size_t>(idx)];
  }
  return ids;
}

int BlockTable::chain_len(int head) const {
  return static_cast<int>(walk(head).size());
}

int BlockTable::expand_chain(int tail) {
  require_allocated(tail);
  if (next_ptr_[static_cast<size_t>(tail)] != -1) {
    fail("block " + std::to_string(tail) + " is not a chain tail");
  }
  if (free_list_.empty()) {
    fail("No free blocks available");  // state unchanged
  }
  const int new_idx = allocate_block();
  next_ptr_[static_cast<size_t>(tail)] = new_idx;
  return new_idx;
}

int BlockTable::resolve_block(int head, int logical_idx) const {
  const std::vector<int> row = walk(head);
  if (logical_idx < 0 || logical_idx >= static_cast<int>(row.size())) {
    fail("logical block " + std::to_string(logical_idx) +
         " out of range for chain of " + std::to_string(row.size()) +
         " block(s)");
  }
  return row[static_cast<size_t>(logical_idx)];
}

std::vector<int> BlockTable::page_table_row(int head, int pad_to) const {
  std::vector<int> row = walk(head);
  if (pad_to >= 0) {
    if (pad_to < static_cast<int>(row.size())) {
      fail("pad_to shorter than chain (" + std::to_string(row.size()) +
           " blocks)");
    }
    row.resize(static_cast<size_t>(pad_to), kPadId);
  }
  return row;
}

}  // namespace ouroboros
