#ifndef CYTNX_LINALG_MDSPAN_HPP_
#define CYTNX_LINALG_MDSPAN_HPP_

#include "TensorT.hpp"
#include "TensorT_traits.hpp"
#include "backend/lapack_mdspan.hpp"
#include "cytnx_error.hpp"
#include "mdspan_concepts.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <variant>

#ifndef BACKEND_TORCH

namespace cytnx {

  namespace linalg_mdspan_detail {

    template <class View, class = void>
    struct access_policy {
      using type = host_access;
    };

    template <class View>
    struct access_policy<View, std::void_t<typename View::access_type>> {
      using type = typename View::access_type;
    };

    template <class View>
    concept HostView = std::same_as<typename access_policy<View>::type, host_access>;

    template <class T>
    bool is_finite_scalar(const T &value) {
      if constexpr (lapack::ComplexLapackScalar<T>) {
        return std::isfinite(value.real()) && std::isfinite(value.imag());
      } else {
        return std::isfinite(value);
      }
    }

    template <mdspan_concepts::MdspanView View>
    std::size_t nonfinite_count(View view) {
      static_assert(View::rank() == 1 || View::rank() == 2,
                    "linear algebra diagnostics support vector and matrix views");
      std::size_t count = 0;
      if constexpr (View::rank() == 1) {
        for (std::size_t i = 0; i < view.extent(0); ++i) {
          if (!is_finite_scalar(view(i))) ++count;
        }
      } else {
        for (std::size_t i = 0; i < view.extent(0); ++i) {
          for (std::size_t j = 0; j < view.extent(1); ++j) {
            if (!is_finite_scalar(view(i, j))) ++count;
          }
        }
      }
      return count;
    }

    template <mdspan_concepts::MdspanView... Views>
    void check_lapack_info(const char *routine, int info, Views... views) {
      if (info == 0) return;
      const std::size_t count = (std::size_t{0} + ... + nonfinite_count(views));
      cytnx_error_msg(
        true,
        "[ERROR] LAPACK %s failed with info = %d. Post-call diagnostic found %llu NaN/Inf "
        "entries in checked arrays.%s",
        routine, info, static_cast<unsigned long long>(count), "\n");
    }

    template <class T>
    struct is_variant : std::false_type {};

    template <class... Alternatives>
    struct is_variant<std::variant<Alternatives...>> : std::true_type {};

    template <class T>
    concept Variant = is_variant<std::remove_cvref_t<T>>::value;

    template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector>
      requires HostView<Matrix> && HostView<Vector> &&
               lapack::SameElementType<Vector, lapack::RealElementOf<Matrix>>
    void svd_values_impl(Matrix a, Vector s) {
      check_lapack_info("gesvd", lapack::gesvd(a, s), a, s);
    }

    template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector,
              lapack::LapackMatrix LeftSingularVectors, lapack::LapackMatrix RightSingularVectors>
      requires HostView<Matrix> && HostView<Vector> && HostView<LeftSingularVectors> &&
               HostView<RightSingularVectors> &&
               lapack::SameElementType<Vector, lapack::RealElementOf<Matrix>> &&
               lapack::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
    void svd_impl(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
      check_lapack_info("gesvd", lapack::gesvd(a, s, u, vt), a, s, u, vt);
    }

    template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector>
      requires HostView<Matrix> && HostView<Vector> &&
               lapack::SameElementType<Vector, lapack::RealElementOf<Matrix>>
    void self_adjoint_eigh_impl(char jobz, char uplo, Matrix a, Vector w) {
      check_lapack_info("eigh", lapack::eigh(jobz, uplo, a, w), a, w);
    }

  }  // namespace linalg_mdspan_detail

  /**
   * @brief Compute singular values of a host layout-right matrix view.
   *
   * This is the checked, algorithm-facing wrapper. The lower LAPACK backend returns `info`; this
   * function treats nonzero `info` as unrecoverable and emits diagnostics before throwing.
   */
  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector>
    requires requires(Matrix a, Vector s) { linalg_mdspan_detail::svd_values_impl(a, s); }
  void svd_values(Matrix a, Vector s) {
    linalg_mdspan_detail::svd_values_impl(a, s);
  }

  /// Variant-lifted overload of `svd_values`.
  template <linalg_mdspan_detail::Variant MatrixVariant,
            linalg_mdspan_detail::Variant VectorVariant>
  void svd_values(MatrixVariant &a, VectorVariant &s) {
    std::visit(
      [](auto &matrix, auto &values) {
        if constexpr (requires { linalg_mdspan_detail::svd_values_impl(matrix, values); }) {
          linalg_mdspan_detail::svd_values_impl(matrix, values);
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
   * @brief Compute the thin singular value decomposition of a host layout-right matrix view.
   *
   * `a` is overwritten by the backend. The output views must have shapes compatible with the thin
   * SVD: `s(min(m,n))`, `u(m,min(m,n))`, and `vt(min(m,n),n)` in logical row-major order.
   */
  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector,
            lapack::LapackMatrix LeftSingularVectors, lapack::LapackMatrix RightSingularVectors>
    requires requires(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
      linalg_mdspan_detail::svd_impl(a, s, u, vt);
    }
  void svd(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
    linalg_mdspan_detail::svd_impl(a, s, u, vt);
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
        if constexpr (requires { linalg_mdspan_detail::svd_impl(matrix, values, left, right); }) {
          linalg_mdspan_detail::svd_impl(matrix, values, left, right);
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
   * @brief Diagonalize a real symmetric or complex Hermitian host layout-right matrix view.
   *
   * The wrapper uses row-major logical indexing and translates the triangular `uplo` selector for
   * the column-major LAPACK backend.
   */
  template <lapack::LapackMatrix Matrix, lapack::RealLapackVector Vector>
    requires requires(char jobz, char uplo, Matrix a, Vector w) {
      linalg_mdspan_detail::self_adjoint_eigh_impl(jobz, uplo, a, w);
    }
  void self_adjoint_eigh(char jobz, char uplo, Matrix a, Vector w) {
    linalg_mdspan_detail::self_adjoint_eigh_impl(jobz, uplo, a, w);
  }

  /// Variant-lifted overload of `self_adjoint_eigh`.
  template <linalg_mdspan_detail::Variant MatrixVariant,
            linalg_mdspan_detail::Variant VectorVariant>
  void self_adjoint_eigh(char jobz, char uplo, MatrixVariant &a, VectorVariant &w) {
    std::visit(
      [jobz, uplo](auto &matrix, auto &values) {
        if constexpr (requires {
                        linalg_mdspan_detail::self_adjoint_eigh_impl(jobz, uplo, matrix, values);
                      }) {
          linalg_mdspan_detail::self_adjoint_eigh_impl(jobz, uplo, matrix, values);
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
