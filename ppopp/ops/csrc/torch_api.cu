#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include "qbmatmul.cu"


namespace ppopp {

torch::stable::Tensor qbmatmul_cuda(const torch::stable::Tensor &a,
                                    const torch::stable::Tensor &b){

}





// Defines the operators
STABLE_TORCH_LIBRARY(ppopp, m) {
  m.def("qbmatmul(Tensor a, Tensor b) -> Tensor");
}

// Registers implementations
STABLE_TORCH_LIBRARY_IMPL(ppopp, CUDA, m) {
  m.impl("qbmatmul", TORCH_BOX(&qbmatmul_cuda));
}

}


