#ifndef CYTNX_BACKEND_LINALG_MDSPAN_HOST_HPP_
#define CYTNX_BACKEND_LINALG_MDSPAN_HOST_HPP_

#include "backend/lapack_mdspan.hpp"

#include <utility>

#ifndef BACKEND_TORCH

namespace cytnx::linalg_mdspan_backend {

  template <class Matrix, class Vector>
    requires requires(Matrix &&matrix, Vector &&values) {
      lapack::svd_values(std::forward<Matrix>(matrix), std::forward<Vector>(values));
    }
  void svd_values(Matrix &&matrix, Vector &&values) {
    lapack::svd_values(std::forward<Matrix>(matrix), std::forward<Vector>(values));
  }

  template <class Matrix, class Vector, class LeftSingularVectors, class RightSingularVectors>
    requires requires(Matrix &&matrix, Vector &&values, LeftSingularVectors &&left,
                      RightSingularVectors &&right) {
      lapack::svd(std::forward<Matrix>(matrix), std::forward<Vector>(values),
                  std::forward<LeftSingularVectors>(left),
                  std::forward<RightSingularVectors>(right));
    }
  void svd(Matrix &&matrix, Vector &&values, LeftSingularVectors &&left,
           RightSingularVectors &&right) {
    lapack::svd(std::forward<Matrix>(matrix), std::forward<Vector>(values),
                std::forward<LeftSingularVectors>(left), std::forward<RightSingularVectors>(right));
  }

  template <class Matrix, class Vector>
    requires requires(char jobz, char uplo, Matrix &&matrix, Vector &&values) {
      lapack::self_adjoint_eigh(jobz, uplo, std::forward<Matrix>(matrix),
                                std::forward<Vector>(values));
    }
  void self_adjoint_eigh(char jobz, char uplo, Matrix &&matrix, Vector &&values) {
    lapack::self_adjoint_eigh(jobz, uplo, std::forward<Matrix>(matrix),
                              std::forward<Vector>(values));
  }

}  // namespace cytnx::linalg_mdspan_backend

#endif  // BACKEND_TORCH

#endif  // CYTNX_BACKEND_LINALG_MDSPAN_HOST_HPP_
