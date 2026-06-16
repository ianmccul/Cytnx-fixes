#ifndef CYTNX_LINALG_MDSPAN_HPP_
#define CYTNX_LINALG_MDSPAN_HPP_

#include "TensorT.hpp"
#include "TensorT_traits.hpp"
#include "backend/linalg_mdspan_host.hpp"
#include "cytnx_error.hpp"
#include "kernel.hpp"

#ifdef UNI_GPU
  #include "backend/linalg_mdspan_cuda.hpp"
#endif

#include <utility>

#ifndef BACKEND_TORCH

namespace cytnx {

  /**
   * @brief Compute singular values of a host layout-right mdspan matrix view.
   *
   * Concrete mdspan arguments call the checked LAPACK backend directly. Variant arguments dispatch
   * over alternatives and report an error if the active alternatives are incompatible.
   */
  template <class MatrixArg, class VectorArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::svd_values_kernel, MatrixArg, VectorArg>
  void svd_values(MatrixArg &&a, VectorArg &&s) {
    invoke_kernel(linalg_mdspan_backend::svd_values_kernel{}, std::forward<MatrixArg>(a),
                  std::forward<VectorArg>(s));
  }

  /**
   * @brief Compute the thin singular value decomposition of a host layout-right mdspan matrix view.
   *
   * `a` is overwritten by the backend. Concrete mdspan arguments call the checked LAPACK backend
   * directly. Variant arguments dispatch over alternatives and report an error if the active
   * alternatives are incompatible.
   */
  template <class MatrixArg, class VectorArg, class LeftSingularVectorsArg,
            class RightSingularVectorsArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::svd_kernel, MatrixArg, VectorArg,
                                  LeftSingularVectorsArg, RightSingularVectorsArg>
  void svd(MatrixArg &&a, VectorArg &&s, LeftSingularVectorsArg &&u, RightSingularVectorsArg &&vt) {
    invoke_kernel(linalg_mdspan_backend::svd_kernel{}, std::forward<MatrixArg>(a),
                  std::forward<VectorArg>(s), std::forward<LeftSingularVectorsArg>(u),
                  std::forward<RightSingularVectorsArg>(vt));
  }

  /**
   * @brief Diagonalize a real symmetric or complex Hermitian host layout-right mdspan matrix view.
   *
   * The wrapper uses row-major logical indexing and translates the triangular `uplo` selector for
   * the column-major LAPACK backend. Variant arguments dispatch over alternatives and report an
   * error if the active alternatives are incompatible.
   */
  template <class MatrixArg, class VectorArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::self_adjoint_eigh_kernel, MatrixArg,
                                  VectorArg>
  void self_adjoint_eigh(char jobz, char uplo, MatrixArg &&a, VectorArg &&w) {
    invoke_kernel(linalg_mdspan_backend::self_adjoint_eigh_kernel{jobz, uplo},
                  std::forward<MatrixArg>(a), std::forward<VectorArg>(w));
  }

  /**
   * @brief Compute eigenvalues of a general square host layout-right mdspan matrix view.
   *
   * Real inputs may have complex eigenvalues, so the eigenvalue output is always complex. Variant
   * arguments dispatch over alternatives and report an error if the active alternatives are
   * incompatible.
   */
  template <class MatrixArg, class VectorArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::eig_values_kernel, MatrixArg, VectorArg>
  void eig_values(MatrixArg &&a, VectorArg &&w) {
    invoke_kernel(linalg_mdspan_backend::eig_values_kernel{}, std::forward<MatrixArg>(a),
                  std::forward<VectorArg>(w));
  }

  /**
   * @brief Compute eigenvalues of a real symmetric tridiagonal matrix.
   *
   * `diagonal` is overwritten with eigenvalues. `offdiagonal` is LAPACK workspace and is not
   * preserved. Variant arguments dispatch over alternatives and report an error if the active
   * alternatives are incompatible.
   */
  template <class DiagonalArg, class OffDiagonalArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::symmetric_tridiagonal_eigh_values_kernel,
                                  DiagonalArg, OffDiagonalArg>
  void symmetric_tridiagonal_eigh_values(DiagonalArg &&diagonal, OffDiagonalArg &&offdiagonal) {
    invoke_kernel(linalg_mdspan_backend::symmetric_tridiagonal_eigh_values_kernel{},
                  std::forward<DiagonalArg>(diagonal), std::forward<OffDiagonalArg>(offdiagonal));
  }

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_LINALG_MDSPAN_HPP_
