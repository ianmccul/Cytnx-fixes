#ifndef CYTNX_BACKEND_LINALG_MDSPAN_HOST_HPP_
#define CYTNX_BACKEND_LINALG_MDSPAN_HOST_HPP_

#include "backend/lapack_mdspan.hpp"

#ifndef BACKEND_TORCH

namespace cytnx::linalg_mdspan_backend {

  struct svd_values_kernel {};

  struct svd_kernel {};

  struct self_adjoint_eigh_kernel {
    char jobz = 'N';
    char uplo = 'U';
  };

  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void run_kernel(svd_values_kernel, Matrix matrix, Vector values) {
    lapack::svd_values(matrix, values);
  }

  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector,
            lapack::LapackMatrix LeftSingularVectors, lapack::LapackMatrix RightSingularVectors>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
             mdspan_concepts::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
  void run_kernel(svd_kernel, Matrix matrix, Vector values, LeftSingularVectors left,
                  RightSingularVectors right) {
    lapack::svd(matrix, values, left, right);
  }

  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void run_kernel(self_adjoint_eigh_kernel kernel, Matrix matrix, Vector values) {
    lapack::self_adjoint_eigh(kernel.jobz, kernel.uplo, matrix, values);
  }

}  // namespace cytnx::linalg_mdspan_backend

#endif  // BACKEND_TORCH

#endif  // CYTNX_BACKEND_LINALG_MDSPAN_HOST_HPP_
