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
   * @brief Compute singular values with the divide-and-conquer SVD driver.
   *
   * Concrete mdspan arguments call the checked LAPACK backend directly. Variant arguments dispatch
   * over alternatives and report an error if the active alternatives are incompatible.
   */
  template <class MatrixArg, class VectorArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::svd_divide_conquer_values_kernel,
                                  MatrixArg, VectorArg>
  void svd_divide_conquer_values(MatrixArg &&a, VectorArg &&s) {
    invoke_kernel(linalg_mdspan_backend::svd_divide_conquer_values_kernel{},
                  std::forward<MatrixArg>(a), std::forward<VectorArg>(s));
  }

  /**
   * @brief Compute the thin SVD with the divide-and-conquer SVD driver.
   *
   * `a` is overwritten by the backend. Concrete mdspan arguments call the checked LAPACK backend
   * directly. Variant arguments dispatch over alternatives and report an error if the active
   * alternatives are incompatible.
   */
  template <class MatrixArg, class VectorArg, class LeftSingularVectorsArg,
            class RightSingularVectorsArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::svd_divide_conquer_kernel, MatrixArg,
                                  VectorArg, LeftSingularVectorsArg, RightSingularVectorsArg>
  void svd_divide_conquer(MatrixArg &&a, VectorArg &&s, LeftSingularVectorsArg &&u,
                          RightSingularVectorsArg &&vt) {
    invoke_kernel(linalg_mdspan_backend::svd_divide_conquer_kernel{}, std::forward<MatrixArg>(a),
                  std::forward<VectorArg>(s), std::forward<LeftSingularVectorsArg>(u),
                  std::forward<RightSingularVectorsArg>(vt));
  }

  /**
   * @brief Solve a least-squares problem using the divide-and-conquer SVD driver.
   *
   * `b` must have at least `max(a.extent(0), a.extent(1))` rows. On return, the first
   * `a.extent(1)` rows of `b` contain the solution. `s` receives the singular values of `a`, and
   * `rank` receives the effective numerical rank.
   */
  template <class MatrixArg, class RightHandSideArg, class SingularValuesArg, class Rcond>
    requires AnyDispatchInvocable<linalg_mdspan_backend::least_squares_kernel, MatrixArg,
                                  RightHandSideArg, SingularValuesArg, blas_int &, Rcond>
  void least_squares(MatrixArg &&a, RightHandSideArg &&b, SingularValuesArg &&s, blas_int &rank,
                     Rcond rcond) {
    invoke_kernel(linalg_mdspan_backend::least_squares_kernel{}, std::forward<MatrixArg>(a),
                  std::forward<RightHandSideArg>(b), std::forward<SingularValuesArg>(s), rank,
                  rcond);
  }

  /**
   * @brief Solve a least-squares problem using LAPACK's default rank cutoff.
   */
  template <class MatrixArg, class RightHandSideArg, class SingularValuesArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::least_squares_kernel, MatrixArg,
                                  RightHandSideArg, SingularValuesArg, blas_int &, double>
  void least_squares(MatrixArg &&a, RightHandSideArg &&b, SingularValuesArg &&s, blas_int &rank) {
    least_squares(std::forward<MatrixArg>(a), std::forward<RightHandSideArg>(b),
                  std::forward<SingularValuesArg>(s), rank, -1.0);
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
   * @brief Diagonalize a Hermitian matrix and write eigenvectors as rows.
   *
   * `vectors(i, j)` is component `j` of eigenvector `i`. This row-major convention keeps each
   * eigenvector contiguous in memory.
   */
  template <class MatrixArg, class VectorArg, class EigenvectorsArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::self_adjoint_eigh_vectors_kernel,
                                  MatrixArg, VectorArg, EigenvectorsArg>
  void self_adjoint_eigh_vectors(char uplo, MatrixArg &&a, VectorArg &&w,
                                 EigenvectorsArg &&vectors) {
    invoke_kernel(linalg_mdspan_backend::self_adjoint_eigh_vectors_kernel{uplo},
                  std::forward<MatrixArg>(a), std::forward<VectorArg>(w),
                  std::forward<EigenvectorsArg>(vectors));
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
   * @brief Compute eigenvalues and right eigenvectors of a general square matrix.
   *
   * Real inputs are internally promoted to a complex LAPACK call. Eigenvectors are written as rows:
   * `vectors(i, j)` is component `j` of eigenvector `i`.
   */
  template <class MatrixArg, class VectorArg, class EigenvectorsArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::eig_right_vectors_kernel, MatrixArg,
                                  VectorArg, EigenvectorsArg>
  void eig_right_vectors(MatrixArg &&a, VectorArg &&w, EigenvectorsArg &&vectors) {
    invoke_kernel(linalg_mdspan_backend::eig_right_vectors_kernel{}, std::forward<MatrixArg>(a),
                  std::forward<VectorArg>(w), std::forward<EigenvectorsArg>(vectors));
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

  /**
   * @brief Invert a square matrix in place.
   *
   * Variant arguments dispatch over alternatives and report an error if the active alternative is
   * incompatible.
   */
  template <class MatrixArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::inverse_inplace_kernel, MatrixArg>
  void inverse_inplace(MatrixArg &&a) {
    invoke_kernel(linalg_mdspan_backend::inverse_inplace_kernel{}, std::forward<MatrixArg>(a));
  }

  /**
   * @brief Compute the thin QR factorization of a host layout-right mdspan matrix view.
   *
   * Variant arguments dispatch over alternatives and report an error if the active alternatives are
   * incompatible.
   */
  template <class MatrixArg, class QArg, class RArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::qr_kernel, MatrixArg, QArg, RArg>
  void qr(MatrixArg &&a, QArg &&q, RArg &&r) {
    invoke_kernel(linalg_mdspan_backend::qr_kernel{}, std::forward<MatrixArg>(a),
                  std::forward<QArg>(q), std::forward<RArg>(r));
  }

  /**
   * @brief Compute the thin LQ factorization of a host layout-right mdspan matrix view.
   *
   * Variant arguments dispatch over alternatives and report an error if the active alternatives are
   * incompatible.
   */
  template <class MatrixArg, class LArg, class QArg>
    requires AnyDispatchInvocable<linalg_mdspan_backend::lq_kernel, MatrixArg, LArg, QArg>
  void lq(MatrixArg &&a, LArg &&l, QArg &&q) {
    invoke_kernel(linalg_mdspan_backend::lq_kernel{}, std::forward<MatrixArg>(a),
                  std::forward<LArg>(l), std::forward<QArg>(q));
  }

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_LINALG_MDSPAN_HPP_
