// pybind11 bindings for the production C++ BlockTable (law sys-blocks).
//
// Exposes module `ouroboros_cpp` with a BlockTable class whose observable
// behaviour mirrors src/block_table.py: same allocation order, same
// all-or-nothing chain semantics, same double-free rejection. std::runtime_error
// is translated by pybind11 into Python RuntimeError automatically.
//
// Note on payload access: Python's get_block() hands out the live list, while
// the binding returns a copy plus set_cell/get_cell for mutation — the
// differential parity driver uses cell-level writes on BOTH sides so the two
// implementations observe identical operations.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <vector>

#include "block_table.h"

namespace py = pybind11;

using ouroboros::BlockTable;
using ouroboros::kBlockSize;

PYBIND11_MODULE(ouroboros_cpp, m) {
  m.doc() = "Ouroboros v7.1 production BlockTable (array-backed linked list)";

  m.attr("BLOCK_SIZE") = kBlockSize;
  m.attr("PAD_ID") = ouroboros::kPadId;

  py::class_<BlockTable>(m, "BlockTable")
      .def(py::init<int>(), py::arg("max_blocks") = 1024)
      .def("allocate_block", &BlockTable::allocate_block)
      .def("free_block", &BlockTable::free_block, py::arg("idx"))
      .def(
          "get_block",
          [](const BlockTable& self, int idx) {
            const int* p = self.get_block(idx);
            return std::vector<int>(p, p + kBlockSize);
          },
          py::arg("idx"),
          "Payload of an allocated block (returned as a copy; mutate via "
          "set_cell/get_cell).")
      .def("set_cell", &BlockTable::set_cell, py::arg("idx"), py::arg("offset"),
           py::arg("value"))
      .def("get_cell", &BlockTable::get_cell, py::arg("idx"), py::arg("offset"))
      .def("link", &BlockTable::link, py::arg("prev_idx"), py::arg("next_idx"))
      .def("allocate_chain", &BlockTable::allocate_chain, py::arg("n_blocks"))
      .def("walk", &BlockTable::walk, py::arg("head"))
      .def("chain_len", &BlockTable::chain_len, py::arg("head"))
      .def("expand_chain", &BlockTable::expand_chain, py::arg("tail"))
      .def("resolve_block", &BlockTable::resolve_block, py::arg("head"),
           py::arg("logical_idx"))
      .def(
          "page_table_row",
          [](const BlockTable& self, int head, std::optional<int> pad_to) {
            return self.page_table_row(head, pad_to.value_or(-1));
          },
          py::arg("head"), py::arg("pad_to") = std::nullopt)
      .def("is_allocated", &BlockTable::is_allocated, py::arg("idx"))
      .def_property_readonly("block_size", [](const BlockTable&) { return kBlockSize; })
      .def_property_readonly("max_blocks", &BlockTable::max_blocks)
      .def_property_readonly("num_free", &BlockTable::num_free)
      // Full-array views mirroring the Python attributes' observable state.
      .def_property_readonly("next_ptr", [](const BlockTable& self) {
        std::vector<int> out;
        out.reserve(static_cast<size_t>(self.max_blocks()));
        for (int i = 0; i < self.max_blocks(); ++i) {
          out.push_back(self.is_allocated(i) ? self.next_ptr_value(i) : -1);
        }
        return out;
      })
      .def_property_readonly("blocks", [](const BlockTable& self) {
        std::vector<std::optional<std::vector<int>>> out;
        out.reserve(static_cast<size_t>(self.max_blocks()));
        for (int i = 0; i < self.max_blocks(); ++i) {
          if (self.is_allocated(i)) {
            const int* p = self.get_block(i);
            out.emplace_back(std::vector<int>(p, p + kBlockSize));
          } else {
            out.emplace_back(std::nullopt);
          }
        }
        return out;
      });
}
