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

  namespace linalg_mdspan_detail {

    struct call_svd_values {
      template <class Matrix, class Vector>
        requires requires(Matrix &&matrix, Vector &&values) {
          linalg_mdspan_backend::svd_values(std::forward<Matrix>(matrix),
                                            std::forward<Vector>(values));
        }
      void operator()(Matrix &&matrix, Vector &&values) const {
        linalg_mdspan_backend::svd_values(std::forward<Matrix>(matrix),
                                          std::forward<Vector>(values));
      }
    };

    struct call_svd {
      template <class Matrix, class Vector, class LeftSingularVectors, class RightSingularVectors>
        requires requires(Matrix &&matrix, Vector &&values, LeftSingularVectors &&left,
                          RightSingularVectors &&right) {
          linalg_mdspan_backend::svd(std::forward<Matrix>(matrix), std::forward<Vector>(values),
                                     std::forward<LeftSingularVectors>(left),
                                     std::forward<RightSingularVectors>(right));
        }
      void operator()(Matrix &&matrix, Vector &&values, LeftSingularVectors &&left,
                      RightSingularVectors &&right) const {
        linalg_mdspan_backend::svd(std::forward<Matrix>(matrix), std::forward<Vector>(values),
                                   std::forward<LeftSingularVectors>(left),
                                   std::forward<RightSingularVectors>(right));
      }
    };

    struct call_self_adjoint_eigh {
      char jobz = 'N';
      char uplo = 'U';

      template <class Matrix, class Vector>
        requires requires(Matrix &&matrix, Vector &&values) {
          linalg_mdspan_backend::self_adjoint_eigh('N', 'U', std::forward<Matrix>(matrix),
                                                   std::forward<Vector>(values));
        }
      void operator()(Matrix &&matrix, Vector &&values) const {
        linalg_mdspan_backend::self_adjoint_eigh(jobz, uplo, std::forward<Matrix>(matrix),
                                                 std::forward<Vector>(values));
      }
    };

  }  // namespace linalg_mdspan_detail

  /**
   * @brief Compute singular values of a host layout-right mdspan matrix view.
   *
   * Concrete mdspan arguments call the checked LAPACK backend directly. Variant arguments dispatch
   * over alternatives and report an error if the active alternatives are incompatible.
   */
  template <class MatrixArg, class VectorArg>
    requires AnyDispatchInvocable<linalg_mdspan_detail::call_svd_values, MatrixArg, VectorArg>
  void svd_values(MatrixArg &&a, VectorArg &&s) {
    invoke_kernel(
      linalg_mdspan_detail::call_svd_values{},
      "[ERROR] svd_values variant alternatives have incompatible dtype, layout, rank, or backend.",
      std::forward<MatrixArg>(a), std::forward<VectorArg>(s));
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
    requires AnyDispatchInvocable<linalg_mdspan_detail::call_svd, MatrixArg, VectorArg,
                                  LeftSingularVectorsArg, RightSingularVectorsArg>
  void svd(MatrixArg &&a, VectorArg &&s, LeftSingularVectorsArg &&u, RightSingularVectorsArg &&vt) {
    invoke_kernel(
      linalg_mdspan_detail::call_svd{},
      "[ERROR] svd variant alternatives have incompatible dtype, layout, rank, or backend.",
      std::forward<MatrixArg>(a), std::forward<VectorArg>(s),
      std::forward<LeftSingularVectorsArg>(u), std::forward<RightSingularVectorsArg>(vt));
  }

  /**
   * @brief Diagonalize a real symmetric or complex Hermitian host layout-right mdspan matrix view.
   *
   * The wrapper uses row-major logical indexing and translates the triangular `uplo` selector for
   * the column-major LAPACK backend. Variant arguments dispatch over alternatives and report an
   * error if the active alternatives are incompatible.
   */
  template <class MatrixArg, class VectorArg>
    requires AnyDispatchInvocable<linalg_mdspan_detail::call_self_adjoint_eigh, MatrixArg,
                                  VectorArg>
  void self_adjoint_eigh(char jobz, char uplo, MatrixArg &&a, VectorArg &&w) {
    invoke_kernel(
      linalg_mdspan_detail::call_self_adjoint_eigh{jobz, uplo},
      "[ERROR] self_adjoint_eigh variant alternatives have incompatible dtype, layout, rank, or "
      "backend.",
      std::forward<MatrixArg>(a), std::forward<VectorArg>(w));
  }

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_LINALG_MDSPAN_HPP_
