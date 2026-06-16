#ifndef CYTNX_LINALG_MDSPAN_HPP_
#define CYTNX_LINALG_MDSPAN_HPP_

#include "TensorT.hpp"
#include "TensorT_traits.hpp"
#include "backend/lapack_mdspan.hpp"
#include "cytnx_error.hpp"

#include <type_traits>

#ifndef BACKEND_TORCH

namespace cytnx {

  /**
   * @brief Compute singular values of a host layout-right mdspan matrix view.
   *
   * Host-accessible mdspan views call the checked LAPACK backend. CUDA overloads can be added here
   * without exposing CUDA details to callers.
   */
  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void svd_values(Matrix a, Vector s) {
    lapack::svd_values(a, s);
  }

  /**
   * @brief Compute the thin singular value decomposition of a host layout-right mdspan matrix view.
   *
   * `a` is overwritten by the backend. Host-accessible mdspan views call the checked LAPACK
   * backend; CUDA overloads can be added here with the same contract.
   */
  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector,
            lapack::LapackMatrix LeftSingularVectors, lapack::LapackMatrix RightSingularVectors>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
             mdspan_concepts::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
  void svd(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
    lapack::svd(a, s, u, vt);
  }

  /**
   * @brief Diagonalize a real symmetric or complex Hermitian host layout-right TensorT matrix view.
   *
   * The wrapper uses row-major logical indexing and translates the triangular `uplo` selector for
   * the column-major LAPACK backend.
   */
  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void self_adjoint_eigh(char jobz, char uplo, Matrix a, Vector w) {
    lapack::self_adjoint_eigh(jobz, uplo, a, w);
  }

  namespace linalg_mdspan_detail {

    struct call_svd_values {
      template <class Matrix, class Vector>
        requires requires(Matrix &matrix, Vector &values) { cytnx::svd_values(matrix, values); }
      void operator()(Matrix &matrix, Vector &values) const {
        cytnx::svd_values(matrix, values);
      }
    };

    struct call_svd {
      template <class Matrix, class Vector, class LeftSingularVectors, class RightSingularVectors>
        requires requires(Matrix &matrix, Vector &values, LeftSingularVectors &left,
                          RightSingularVectors &right) { cytnx::svd(matrix, values, left, right); }
      void operator()(Matrix &matrix, Vector &values, LeftSingularVectors &left,
                      RightSingularVectors &right) const {
        cytnx::svd(matrix, values, left, right);
      }
    };

    struct call_self_adjoint_eigh {
      template <class Matrix, class Vector>
        requires requires(Matrix &matrix, Vector &values) {
          cytnx::self_adjoint_eigh('N', 'U', matrix, values);
        }
      void operator()(Matrix &matrix, Vector &values) const {
        cytnx::self_adjoint_eigh('N', 'U', matrix, values);
      }

      template <class Matrix, class Vector>
        requires requires(Matrix &matrix, Vector &values) {
          cytnx::self_adjoint_eigh('N', 'U', matrix, values);
        }
      void operator()(char jobz, char uplo, Matrix &matrix, Vector &values) const {
        cytnx::self_adjoint_eigh(jobz, uplo, matrix, values);
      }
    };

  }  // namespace linalg_mdspan_detail

  /// Variant-lifted overload of `svd_values`.
  template <class MatrixArg, class VectorArg>
    requires AnyDispatchInvocable<linalg_mdspan_detail::call_svd_values, MatrixArg, VectorArg>
  void svd_values(MatrixArg &a, VectorArg &s) {
    tensor_t_detail::dispatch_visit(
      [](auto &matrix, auto &values) {
        if constexpr (std::is_invocable_v<linalg_mdspan_detail::call_svd_values, decltype(matrix),
                                          decltype(values)>) {
          linalg_mdspan_detail::call_svd_values{}(matrix, values);
        } else {
          cytnx_error_msg(true,
                          "[ERROR] svd_values variant alternatives have incompatible dtype, "
                          "layout, rank, or backend.%s",
                          "\n");
        }
      },
      a, s);
  }

  /// Variant-lifted overload of `svd`.
  template <class MatrixArg, class VectorArg, class LeftSingularVectorsArg,
            class RightSingularVectorsArg>
    requires AnyDispatchInvocable<linalg_mdspan_detail::call_svd, MatrixArg, VectorArg,
                                  LeftSingularVectorsArg, RightSingularVectorsArg>
  void svd(MatrixArg &a, VectorArg &s, LeftSingularVectorsArg &u, RightSingularVectorsArg &vt) {
    tensor_t_detail::dispatch_visit(
      [](auto &matrix, auto &values, auto &left, auto &right) {
        if constexpr (std::is_invocable_v<linalg_mdspan_detail::call_svd, decltype(matrix),
                                          decltype(values), decltype(left), decltype(right)>) {
          linalg_mdspan_detail::call_svd{}(matrix, values, left, right);
        } else {
          cytnx_error_msg(true,
                          "[ERROR] svd variant alternatives have incompatible dtype, layout, rank, "
                          "or backend.%s",
                          "\n");
        }
      },
      a, s, u, vt);
  }

  /// Variant-lifted overload of `self_adjoint_eigh`.
  template <class MatrixArg, class VectorArg>
    requires AnyDispatchInvocable<linalg_mdspan_detail::call_self_adjoint_eigh, MatrixArg,
                                  VectorArg>
  void self_adjoint_eigh(char jobz, char uplo, MatrixArg &a, VectorArg &w) {
    tensor_t_detail::dispatch_visit(
      [jobz, uplo](auto &matrix, auto &values) {
        if constexpr (std::is_invocable_v<linalg_mdspan_detail::call_self_adjoint_eigh,
                                          decltype(matrix), decltype(values)>) {
          linalg_mdspan_detail::call_self_adjoint_eigh{}(jobz, uplo, matrix, values);
        } else {
          cytnx_error_msg(true,
                          "[ERROR] self_adjoint_eigh variant alternatives have incompatible dtype, "
                          "layout, rank, or backend.%s",
                          "\n");
        }
      },
      a, w);
  }

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_LINALG_MDSPAN_HPP_
