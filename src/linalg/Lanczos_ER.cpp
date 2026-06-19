#include "linalg.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <stdexcept>

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace linalg {

    std::vector<Tensor> Lanczos_ER(LinOp *Hop, cytnx_uint64 k, bool is_V, cytnx_uint64 maxiter,
                                   double CvgCrit, bool is_row, const Tensor &Tin,
                                   cytnx_uint32 max_krydim, bool verbose) {
      (void)Hop;
      (void)k;
      (void)is_V;
      (void)maxiter;
      (void)CvgCrit;
      (void)is_row;
      (void)Tin;
      (void)max_krydim;
      (void)verbose;

      throw std::runtime_error(
        "[ERROR][Lanczos_ER] The old explicitly restarted Lanczos implementation has been disabled "
        "because it is numerically incorrect. For general Hermitian eigenvalue problems, use the "
        "ARPACK-backed cytnx.linalg.Lanczos(..., which=\"SA\") entry point for the smallest "
        "algebraic eigenvalue, or choose another ARPACK 'which' selector as needed. The "
        "cytnx.linalg.Lanczos(..., method=\"Gnd\") path is a non-restarted Lanczos routine "
        "intended for specialized local ground-state solves where bounded matvec count is more "
        "important than standalone eigensolver semantics.");
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
