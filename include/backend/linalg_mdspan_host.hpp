#ifndef CYTNX_BACKEND_LINALG_MDSPAN_HOST_HPP_
#define CYTNX_BACKEND_LINALG_MDSPAN_HOST_HPP_

#include "backend/lapack_mdspan.hpp"

#include <string_view>

#ifndef BACKEND_TORCH

namespace cytnx::linalg_mdspan_backend {

  struct svd_values_kernel {
    static constexpr std::string_view name = "svd_values";
  };

  struct svd_kernel {
    static constexpr std::string_view name = "svd";
  };

  struct svd_divide_conquer_values_kernel {
    static constexpr std::string_view name = "svd_divide_conquer_values";
  };

  struct svd_divide_conquer_kernel {
    static constexpr std::string_view name = "svd_divide_conquer";
  };

  struct self_adjoint_eigh_kernel {
    static constexpr std::string_view name = "self_adjoint_eigh";
    char jobz = 'N';
    char uplo = 'U';
  };

  struct eig_values_kernel {
    static constexpr std::string_view name = "eig_values";
  };

  struct symmetric_tridiagonal_eigh_values_kernel {
    static constexpr std::string_view name = "symmetric_tridiagonal_eigh_values";
  };

  struct inverse_inplace_kernel {
    static constexpr std::string_view name = "inverse_inplace";
  };

  struct qr_kernel {
    static constexpr std::string_view name = "qr";
  };

  struct lq_kernel {
    static constexpr std::string_view name = "lq";
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
  void run_kernel(svd_divide_conquer_values_kernel, Matrix matrix, Vector values) {
    lapack::svd_divide_conquer_values(matrix, values);
  }

  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector,
            lapack::LapackMatrix LeftSingularVectors, lapack::LapackMatrix RightSingularVectors>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
             mdspan_concepts::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
  void run_kernel(svd_divide_conquer_kernel, Matrix matrix, Vector values, LeftSingularVectors left,
                  RightSingularVectors right) {
    lapack::svd_divide_conquer(matrix, values, left, right);
  }

  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void run_kernel(self_adjoint_eigh_kernel kernel, Matrix matrix, Vector values) {
    lapack::self_adjoint_eigh(kernel.jobz, kernel.uplo, matrix, values);
  }

  template <lapack::LapackMatrix Matrix, lapack::ComplexLapackVector Vector>
    requires lapack::LapackEigenvalueVector<Matrix, Vector>
  void run_kernel(eig_values_kernel, Matrix matrix, Vector values) {
    lapack::eig_values(matrix, values);
  }

  template <lapack::RealLapackVector Diagonal, lapack::RealLapackVector OffDiagonal>
    requires mdspan_concepts::SameElementType<Diagonal, OffDiagonal>
  void run_kernel(symmetric_tridiagonal_eigh_values_kernel, Diagonal diagonal,
                  OffDiagonal offdiagonal) {
    lapack::symmetric_tridiagonal_eigh_values(diagonal, offdiagonal);
  }

  template <lapack::LapackMatrix Matrix>
  void run_kernel(inverse_inplace_kernel, Matrix matrix) {
    lapack::inverse_inplace(matrix);
  }

  template <lapack::RealLapackMatrix Matrix, lapack::RealLapackMatrix QMatrix,
            lapack::RealLapackMatrix RMatrix>
    requires mdspan_concepts::SameElementType<Matrix, QMatrix, RMatrix>
  void run_kernel(qr_kernel, Matrix matrix, QMatrix q, RMatrix r) {
    lapack::qr(matrix, q, r);
  }

  template <lapack::RealLapackMatrix Matrix, lapack::RealLapackMatrix LMatrix,
            lapack::RealLapackMatrix QMatrix>
    requires mdspan_concepts::SameElementType<Matrix, LMatrix, QMatrix>
  void run_kernel(lq_kernel, Matrix matrix, LMatrix l, QMatrix q) {
    lapack::lq(matrix, l, q);
  }

}  // namespace cytnx::linalg_mdspan_backend

#endif  // BACKEND_TORCH

#endif  // CYTNX_BACKEND_LINALG_MDSPAN_HOST_HPP_
