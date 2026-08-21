#include <torch/extension.h>

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/pytorch_context.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/torch_util.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Exchange plain tensors with Python.  k2's RaggedArc and DenseFsaVec
  // pybind classes are module-local, so a separately linked extension cannot
  // safely accept those Python wrapper objects directly.
  m.def(
      "intersect_dense_factorized",
      [](torch::Tensor arc_row_splits1, torch::Tensor arc_row_splits2,
         torch::Tensor arc_values, torch::Tensor token_row_splits,
         torch::Tensor token_scores, torch::Tensor speaker_scores,
         int32_t token_vocab_size, float output_beam, int32_t max_states,
         int32_t max_arcs) {
        k2::DeviceGuard guard(k2::GetContext(token_scores));
        auto rs1 = k2::FromTorch<int32_t>(arc_row_splits1);
        auto rs2 = k2::FromTorch<int32_t>(arc_row_splits2);
        auto arcs = k2::FromTorch<k2::Arc>(arc_values);
        k2::RaggedShape arc_shape = k2::RaggedShape3(
            &rs1, nullptr, -1, &rs2, nullptr, arcs.Dim());
        k2::FsaVec a_fsas(arc_shape, arcs);

        auto token_rs = k2::FromTorch<int32_t>(token_row_splits);
        auto token_array =
            k2::FromTorch<float>(token_scores, k2::Array2Tag{});
        k2::RaggedShape token_shape =
            k2::RaggedShape2(&token_rs, nullptr, token_array.Dim0());
        k2::DenseFsaVec token_fsas(token_shape, token_array);
        auto speakers =
            k2::FromTorch<float>(speaker_scores, k2::Array2Tag{});

        k2::Array1<int32_t> map =
            k2::Arange(a_fsas.Context(), 0, a_fsas.Dim0());
        k2::Array1<int32_t> map_a, map_token, map_speaker;
        k2::FsaVec out;
        k2::IntersectDenseFactorized(
            a_fsas, token_fsas, speakers, token_vocab_size, &map, output_beam,
            max_states, max_arcs, &out, &map_a, &map_token, &map_speaker);
        k2::Tensor out_tensor = k2::FsaVecToTensor(out);
        return py::make_tuple(k2::ToTorch(out_tensor),
                              k2::ToTorch(map_a), k2::ToTorch(map_token),
                              k2::ToTorch(map_speaker));
      },
      py::arg("arc_row_splits1"), py::arg("arc_row_splits2"),
      py::arg("arc_values"), py::arg("token_row_splits"),
      py::arg("token_scores"), py::arg("speaker_scores"),
      py::arg("token_vocab_size"), py::arg("output_beam"),
      py::arg("max_states") = 15000000,
      py::arg("max_arcs") = 1073741824);
}
