#ifndef CYTNX_BACKEND_LAPACK_MDSPAN_HPP_
#define CYTNX_BACKEND_LAPACK_MDSPAN_HPP_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "mdspan.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#ifndef BACKEND_TORCH

extern "C" {
void sgesvd_(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n, float *a,
             const blas_int *lda, float *s, float *u, const blas_int *ldu, float *vt,
             const blas_int *ldvt, float *work, const blas_int *lwork, blas_int *info);
void dgesvd_(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n, double *a,
             const blas_int *lda, double *s, double *u, const blas_int *ldu, double *vt,
             const blas_int *ldvt, double *work, const blas_int *lwork, blas_int *info);
void cgesvd_(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
             std::complex<float> *a, const blas_int *lda, float *s, std::complex<float> *u,
             const blas_int *ldu, std::complex<float> *vt, const blas_int *ldvt,
             std::complex<float> *work, const blas_int *lwork, float *rwork, blas_int *info);
void zgesvd_(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
             std::complex<double> *a, const blas_int *lda, double *s, std::complex<double> *u,
             const blas_int *ldu, std::complex<double> *vt, const blas_int *ldvt,
             std::complex<double> *work, const blas_int *lwork, double *rwork, blas_int *info);

void ssyev_(const char *jobz, const char *uplo, const blas_int *n, float *a, const blas_int *lda,
            float *w, float *work, const blas_int *lwork, blas_int *info);
void dsyev_(const char *jobz, const char *uplo, const blas_int *n, double *a, const blas_int *lda,
            double *w, double *work, const blas_int *lwork, blas_int *info);
void cheev_(const char *jobz, const char *uplo, const blas_int *n, std::complex<float> *a,
            const blas_int *lda, float *w, std::complex<float> *work, const blas_int *lwork,
            float *rwork, blas_int *info);
void zheev_(const char *jobz, const char *uplo, const blas_int *n, std::complex<double> *a,
            const blas_int *lda, double *w, std::complex<double> *work, const blas_int *lwork,
            double *rwork, blas_int *info);
}

namespace cytnx::lapack {

  namespace detail {

    template <class T>
    struct real_scalar {
      using type = T;
    };

    template <class T>
    struct real_scalar<std::complex<T>> {
      using type = T;
    };

    template <class T>
    using real_scalar_t = typename real_scalar<std::remove_cv_t<T>>::type;

    template <class T>
    constexpr bool is_lapack_scalar_v =
      std::is_same_v<std::remove_cv_t<T>, float> || std::is_same_v<std::remove_cv_t<T>, double> ||
      std::is_same_v<std::remove_cv_t<T>, std::complex<float>> ||
      std::is_same_v<std::remove_cv_t<T>, std::complex<double>>;

    template <class View>
    constexpr bool is_layout_right_view_v =
      std::is_same_v<typename View::layout_type, stdex::layout_right>;

    template <class View>
    constexpr bool is_lapack_vector_v = View::rank() == 1 && is_layout_right_view_v<View>;

    template <class View>
    constexpr bool is_lapack_matrix_v = View::rank() == 2 && is_layout_right_view_v<View>;

    inline blas_int to_blas_int(std::size_t value, const char *name) {
      cytnx_error_msg(value > static_cast<std::size_t>(std::numeric_limits<blas_int>::max()),
                      "[ERROR] LAPACK dimension %s exceeds blas_int range.%s", name, "\n");
      return static_cast<blas_int>(value);
    }

    inline char transpose_uplo(char uplo) {
      if (uplo == 'U' || uplo == 'u') return 'L';
      if (uplo == 'L' || uplo == 'l') return 'U';
      cytnx_error_msg(true, "[ERROR] LAPACK uplo must be 'U' or 'L'.%s", "\n");
      return uplo;
    }

  }  // namespace detail

  namespace native {

    inline void gesvd(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
                      float *a, const blas_int *lda, float *s, float *u, const blas_int *ldu,
                      float *vt, const blas_int *ldvt, float *work, const blas_int *lwork,
                      blas_int *info) {
      sgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    inline void gesvd(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
                      double *a, const blas_int *lda, double *s, double *u, const blas_int *ldu,
                      double *vt, const blas_int *ldvt, double *work, const blas_int *lwork,
                      blas_int *info) {
      dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    inline void gesvd(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
                      std::complex<float> *a, const blas_int *lda, float *s, std::complex<float> *u,
                      const blas_int *ldu, std::complex<float> *vt, const blas_int *ldvt,
                      std::complex<float> *work, const blas_int *lwork, float *rwork,
                      blas_int *info) {
      cgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
    }

    inline void gesvd(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
                      std::complex<double> *a, const blas_int *lda, double *s,
                      std::complex<double> *u, const blas_int *ldu, std::complex<double> *vt,
                      const blas_int *ldvt, std::complex<double> *work, const blas_int *lwork,
                      double *rwork, blas_int *info) {
      zgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
    }

    inline void syev(const char *jobz, const char *uplo, const blas_int *n, float *a,
                     const blas_int *lda, float *w, float *work, const blas_int *lwork,
                     blas_int *info) {
      ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
    }

    inline void syev(const char *jobz, const char *uplo, const blas_int *n, double *a,
                     const blas_int *lda, double *w, double *work, const blas_int *lwork,
                     blas_int *info) {
      dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
    }

    inline void heev(const char *jobz, const char *uplo, const blas_int *n, std::complex<float> *a,
                     const blas_int *lda, float *w, std::complex<float> *work,
                     const blas_int *lwork, float *rwork, blas_int *info) {
      cheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
    }

    inline void heev(const char *jobz, const char *uplo, const blas_int *n, std::complex<double> *a,
                     const blas_int *lda, double *w, std::complex<double> *work,
                     const blas_int *lwork, double *rwork, blas_int *info) {
      zheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
    }

  }  // namespace native

  namespace row_major {

    template <class Matrix, class Vector>
    int gesvd(Matrix a, Vector s) {
      using scalar_type = typename Matrix::element_type;
      using real_type = detail::real_scalar_t<scalar_type>;
      static_assert(detail::is_lapack_matrix_v<Matrix>);
      static_assert(detail::is_lapack_vector_v<Vector>);
      static_assert(detail::is_lapack_scalar_v<scalar_type>);
      static_assert(std::is_same_v<typename Vector::element_type, real_type>);

      const auto rows = a.extent(0);
      const auto cols = a.extent(1);
      const auto min_dim = std::min(rows, cols);
      cytnx_error_msg(s.extent(0) < min_dim,
                      "[ERROR] LAPACK gesvd singular-value output is too small.%s", "\n");

      const blas_int native_m = detail::to_blas_int(cols, "cols");
      const blas_int native_n = detail::to_blas_int(rows, "rows");
      const blas_int lda = std::max<blas_int>(1, native_m);
      const blas_int one = 1;
      blas_int info = 0;
      blas_int lwork = -1;

      if constexpr (std::is_same_v<scalar_type, float> || std::is_same_v<scalar_type, double>) {
        scalar_type work_query{};
        native::gesvd("N", "N", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      nullptr, &one, nullptr, &one, &work_query, &lwork, &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gesvd("N", "N", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      nullptr, &one, nullptr, &one, work.data(), &lwork, &info);
      } else {
        scalar_type work_query{};
        const std::size_t rwork_size = std::max<std::size_t>(1, 5 * min_dim);
        std::vector<real_type> rwork(rwork_size);
        native::gesvd("N", "N", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      nullptr, &one, nullptr, &one, &work_query, &lwork, rwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gesvd("N", "N", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      nullptr, &one, nullptr, &one, work.data(), &lwork, rwork.data(), &info);
      }

      return info;
    }

    template <class Matrix, class Vector, class LeftSingularVectors, class RightSingularVectors>
    int gesvd(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
      using scalar_type = typename Matrix::element_type;
      using real_type = detail::real_scalar_t<scalar_type>;
      static_assert(detail::is_lapack_matrix_v<Matrix>);
      static_assert(detail::is_lapack_vector_v<Vector>);
      static_assert(detail::is_lapack_matrix_v<LeftSingularVectors>);
      static_assert(detail::is_lapack_matrix_v<RightSingularVectors>);
      static_assert(detail::is_lapack_scalar_v<scalar_type>);
      static_assert(std::is_same_v<typename Vector::element_type, real_type>);
      static_assert(std::is_same_v<typename LeftSingularVectors::element_type, scalar_type>);
      static_assert(std::is_same_v<typename RightSingularVectors::element_type, scalar_type>);

      const auto rows = a.extent(0);
      const auto cols = a.extent(1);
      const auto min_dim = std::min(rows, cols);
      cytnx_error_msg(s.extent(0) < min_dim,
                      "[ERROR] LAPACK gesvd singular-value output is too small.%s", "\n");
      cytnx_error_msg(u.extent(0) < rows || u.extent(1) < min_dim,
                      "[ERROR] LAPACK gesvd U output has incompatible shape.%s", "\n");
      cytnx_error_msg(vt.extent(0) < min_dim || vt.extent(1) < cols,
                      "[ERROR] LAPACK gesvd VT output has incompatible shape.%s", "\n");

      const blas_int native_m = detail::to_blas_int(cols, "cols");
      const blas_int native_n = detail::to_blas_int(rows, "rows");
      const blas_int lda = std::max<blas_int>(1, native_m);
      const blas_int native_ldu = std::max<blas_int>(1, native_m);
      const blas_int native_ldvt = std::max<blas_int>(1, detail::to_blas_int(min_dim, "min_dim"));
      blas_int info = 0;
      blas_int lwork = -1;

      if constexpr (std::is_same_v<scalar_type, float> || std::is_same_v<scalar_type, double>) {
        scalar_type work_query{};
        native::gesvd("S", "S", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      vt.data_handle(), &native_ldu, u.data_handle(), &native_ldvt, &work_query,
                      &lwork, &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gesvd("S", "S", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      vt.data_handle(), &native_ldu, u.data_handle(), &native_ldvt, work.data(),
                      &lwork, &info);
      } else {
        scalar_type work_query{};
        const std::size_t rwork_size = std::max<std::size_t>(1, 5 * min_dim);
        std::vector<real_type> rwork(rwork_size);
        native::gesvd("S", "S", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      vt.data_handle(), &native_ldu, u.data_handle(), &native_ldvt, &work_query,
                      &lwork, rwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gesvd("S", "S", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      vt.data_handle(), &native_ldu, u.data_handle(), &native_ldvt, work.data(),
                      &lwork, rwork.data(), &info);
      }

      return info;
    }

    template <class Matrix, class Vector>
    int eigh(char jobz, char uplo, Matrix a, Vector w) {
      using scalar_type = typename Matrix::element_type;
      using real_type = detail::real_scalar_t<scalar_type>;
      static_assert(detail::is_lapack_matrix_v<Matrix>);
      static_assert(detail::is_lapack_vector_v<Vector>);
      static_assert(detail::is_lapack_scalar_v<scalar_type>);
      static_assert(std::is_same_v<typename Vector::element_type, real_type>);

      const auto n = a.extent(0);
      cytnx_error_msg(a.extent(1) != n, "[ERROR] LAPACK eigh input must be square.%s", "\n");
      cytnx_error_msg(w.extent(0) < n, "[ERROR] LAPACK eigh eigenvalue output is too small.%s",
                      "\n");

      const blas_int native_n = detail::to_blas_int(n, "n");
      const blas_int lda = std::max<blas_int>(1, native_n);
      const char native_uplo = detail::transpose_uplo(uplo);
      blas_int info = 0;
      blas_int lwork = -1;

      if constexpr (std::is_same_v<scalar_type, float> || std::is_same_v<scalar_type, double>) {
        scalar_type work_query{};
        native::syev(&jobz, &native_uplo, &native_n, a.data_handle(), &lda, w.data_handle(),
                     &work_query, &lwork, &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::syev(&jobz, &native_uplo, &native_n, a.data_handle(), &lda, w.data_handle(),
                     work.data(), &lwork, &info);
      } else {
        scalar_type work_query{};
        const std::size_t rwork_size = std::max<std::size_t>(1, n > 0 ? 3 * n - 2 : 0);
        std::vector<real_type> rwork(rwork_size);
        native::heev(&jobz, &native_uplo, &native_n, a.data_handle(), &lda, w.data_handle(),
                     &work_query, &lwork, rwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::heev(&jobz, &native_uplo, &native_n, a.data_handle(), &lda, w.data_handle(),
                     work.data(), &lwork, rwork.data(), &info);
      }

      return info;
    }

  }  // namespace row_major

}  // namespace cytnx::lapack

#endif  // BACKEND_TORCH

#endif  // CYTNX_BACKEND_LAPACK_MDSPAN_HPP_
