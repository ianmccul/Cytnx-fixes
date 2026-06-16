#ifndef CYTNX_LINALG_MDSPAN_HPP_
#define CYTNX_LINALG_MDSPAN_HPP_

#include "TensorT.hpp"
#include "TensorT_traits.hpp"
#include "backend/lapack_mdspan.hpp"
#include "cytnx_error.hpp"

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <variant>

#ifndef BACKEND_TORCH

namespace cytnx {

  namespace linalg_mdspan_detail {

    template <class T>
    struct is_variant : std::false_type {};

    template <class... Alternatives>
    struct is_variant<std::variant<Alternatives...>> : std::true_type {};

    template <class T>
    concept Variant = is_variant<std::remove_cvref_t<T>>::value;

  }  // namespace linalg_mdspan_detail

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

  /// Variant-lifted overload of `svd_values`.
  template <linalg_mdspan_detail::Variant MatrixVariant,
            linalg_mdspan_detail::Variant VectorVariant>
  void svd_values(MatrixVariant &a, VectorVariant &s) {
    std::visit(
      [](auto &matrix, auto &values) {
        if constexpr (requires { cytnx::svd_values(matrix, values); }) {
          cytnx::svd_values(matrix, values);
        } else {
          cytnx_error_msg(true,
                          "[ERROR] svd_values variant alternatives have incompatible dtype, "
                          "layout, rank, or backend.%s",
                          "\n");
        }
      },
      a, s);
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

  /// Variant-lifted overload of `svd`.
  template <linalg_mdspan_detail::Variant MatrixVariant,
            linalg_mdspan_detail::Variant VectorVariant,
            linalg_mdspan_detail::Variant LeftSingularVectorsVariant,
            linalg_mdspan_detail::Variant RightSingularVectorsVariant>
  void svd(MatrixVariant &a, VectorVariant &s, LeftSingularVectorsVariant &u,
           RightSingularVectorsVariant &vt) {
    std::visit(
      [](auto &matrix, auto &values, auto &left, auto &right) {
        if constexpr (requires { cytnx::svd(matrix, values, left, right); }) {
          cytnx::svd(matrix, values, left, right);
        } else {
          cytnx_error_msg(true,
                          "[ERROR] svd variant alternatives have incompatible dtype, layout, rank, "
                          "or backend.%s",
                          "\n");
        }
      },
      a, s, u, vt);
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

  /// Variant-lifted overload of `self_adjoint_eigh`.
  template <linalg_mdspan_detail::Variant MatrixVariant,
            linalg_mdspan_detail::Variant VectorVariant>
  void self_adjoint_eigh(char jobz, char uplo, MatrixVariant &a, VectorVariant &w) {
    std::visit(
      [jobz, uplo](auto &matrix, auto &values) {
        if constexpr (requires { cytnx::self_adjoint_eigh(jobz, uplo, matrix, values); }) {
          cytnx::self_adjoint_eigh(jobz, uplo, matrix, values);
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
