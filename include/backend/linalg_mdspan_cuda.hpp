#ifndef CYTNX_BACKEND_LINALG_MDSPAN_CUDA_HPP_
#define CYTNX_BACKEND_LINALG_MDSPAN_CUDA_HPP_

#ifndef UNI_GPU
  #error "linalg_mdspan_cuda.hpp requires UNI_GPU"
#endif

#ifndef BACKEND_TORCH

namespace cytnx::linalg_mdspan_backend {

  // CUDA mdspan linalg overloads will live here.

}  // namespace cytnx::linalg_mdspan_backend

#endif  // BACKEND_TORCH

#endif  // CYTNX_BACKEND_LINALG_MDSPAN_CUDA_HPP_
