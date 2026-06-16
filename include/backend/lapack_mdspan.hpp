#ifndef CYTNX_BACKEND_LAPACK_MDSPAN_HPP_
#define CYTNX_BACKEND_LAPACK_MDSPAN_HPP_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "mdspan.hpp"
#include "mdspan_concepts.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#ifndef BACKEND_TORCH

namespace cytnx::lapack::fortran {

  // The raw Fortran ABI is isolated here. LAPACK routines with character arguments may need
  // compiler-specific hidden length handling; this namespace can later be replaced by ISO_C_BINDING
  // shims without changing the mdspan-facing wrappers below.
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

  void sgeev_(const char *jobvl, const char *jobvr, const blas_int *n, float *a,
              const blas_int *lda, float *wr, float *wi, float *vl, const blas_int *ldvl, float *vr,
              const blas_int *ldvr, float *work, const blas_int *lwork, blas_int *info);
  void dgeev_(const char *jobvl, const char *jobvr, const blas_int *n, double *a,
              const blas_int *lda, double *wr, double *wi, double *vl, const blas_int *ldvl,
              double *vr, const blas_int *ldvr, double *work, const blas_int *lwork,
              blas_int *info);
  void cgeev_(const char *jobvl, const char *jobvr, const blas_int *n, std::complex<float> *a,
              const blas_int *lda, std::complex<float> *w, std::complex<float> *vl,
              const blas_int *ldvl, std::complex<float> *vr, const blas_int *ldvr,
              std::complex<float> *work, const blas_int *lwork, float *rwork, blas_int *info);
  void zgeev_(const char *jobvl, const char *jobvr, const blas_int *n, std::complex<double> *a,
              const blas_int *lda, std::complex<double> *w, std::complex<double> *vl,
              const blas_int *ldvl, std::complex<double> *vr, const blas_int *ldvr,
              std::complex<double> *work, const blas_int *lwork, double *rwork, blas_int *info);
  }

}  // namespace cytnx::lapack::fortran

namespace cytnx::lapack {

  namespace detail {

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

  template <class T>
  concept RealLapackScalar =
    std::is_same_v<std::remove_cv_t<T>, float> || std::is_same_v<std::remove_cv_t<T>, double>;

  template <class T>
  concept ComplexLapackScalar = std::is_same_v<std::remove_cv_t<T>, std::complex<float>> ||
                                std::is_same_v<std::remove_cv_t<T>, std::complex<double>>;

  template <class T>
  concept LapackScalar = RealLapackScalar<T> || ComplexLapackScalar<T>;

  template <class View>
  concept LapackVector =
    mdspan_concepts::LayoutRightVector<View> && mdspan_concepts::HostAccessible<View> &&
    LapackScalar<typename View::element_type>;

  template <class View>
  concept RealLapackVector = LapackVector<View> && RealLapackScalar<typename View::element_type>;

  template <class View>
  concept ComplexLapackVector =
    LapackVector<View> && ComplexLapackScalar<typename View::element_type>;

  template <class Matrix, class Vector>
  concept LapackEigenvalueVector =
    ComplexLapackVector<Vector> &&
    ((RealLapackScalar<typename Matrix::element_type> &&
      std::same_as<typename Vector::element_type, std::complex<typename Matrix::element_type>>) ||
     (ComplexLapackScalar<typename Matrix::element_type> &&
      std::same_as<typename Matrix::element_type, typename Vector::element_type>));

  template <class View>
  concept LapackMatrix =
    mdspan_concepts::LayoutRightMatrix<View> && mdspan_concepts::HostAccessible<View> &&
    LapackScalar<typename View::element_type>;

  template <class View>
  concept RealLapackMatrix = LapackMatrix<View> && RealLapackScalar<typename View::element_type>;

  template <class View>
  concept ComplexLapackMatrix =
    LapackMatrix<View> && ComplexLapackScalar<typename View::element_type>;

  namespace lowlevel {

    namespace native {

      inline void gesvd(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
                        float *a, const blas_int *lda, float *s, float *u, const blas_int *ldu,
                        float *vt, const blas_int *ldvt, float *work, const blas_int *lwork,
                        blas_int *info) {
        fortran::sgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
      }

      inline void gesvd(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
                        double *a, const blas_int *lda, double *s, double *u, const blas_int *ldu,
                        double *vt, const blas_int *ldvt, double *work, const blas_int *lwork,
                        blas_int *info) {
        fortran::dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
      }

      inline void gesvd(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
                        std::complex<float> *a, const blas_int *lda, float *s,
                        std::complex<float> *u, const blas_int *ldu, std::complex<float> *vt,
                        const blas_int *ldvt, std::complex<float> *work, const blas_int *lwork,
                        float *rwork, blas_int *info) {
        fortran::cgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
      }

      inline void gesvd(const char *jobu, const char *jobvt, const blas_int *m, const blas_int *n,
                        std::complex<double> *a, const blas_int *lda, double *s,
                        std::complex<double> *u, const blas_int *ldu, std::complex<double> *vt,
                        const blas_int *ldvt, std::complex<double> *work, const blas_int *lwork,
                        double *rwork, blas_int *info) {
        fortran::zgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
      }

      inline void syev(const char *jobz, const char *uplo, const blas_int *n, float *a,
                       const blas_int *lda, float *w, float *work, const blas_int *lwork,
                       blas_int *info) {
        fortran::ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
      }

      inline void syev(const char *jobz, const char *uplo, const blas_int *n, double *a,
                       const blas_int *lda, double *w, double *work, const blas_int *lwork,
                       blas_int *info) {
        fortran::dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
      }

      inline void heev(const char *jobz, const char *uplo, const blas_int *n,
                       std::complex<float> *a, const blas_int *lda, float *w,
                       std::complex<float> *work, const blas_int *lwork, float *rwork,
                       blas_int *info) {
        fortran::cheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
      }

      inline void heev(const char *jobz, const char *uplo, const blas_int *n,
                       std::complex<double> *a, const blas_int *lda, double *w,
                       std::complex<double> *work, const blas_int *lwork, double *rwork,
                       blas_int *info) {
        fortran::zheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
      }

      inline void geev(const char *jobvl, const char *jobvr, const blas_int *n, float *a,
                       const blas_int *lda, float *wr, float *wi, float *vl, const blas_int *ldvl,
                       float *vr, const blas_int *ldvr, float *work, const blas_int *lwork,
                       blas_int *info) {
        fortran::sgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
      }

      inline void geev(const char *jobvl, const char *jobvr, const blas_int *n, double *a,
                       const blas_int *lda, double *wr, double *wi, double *vl,
                       const blas_int *ldvl, double *vr, const blas_int *ldvr, double *work,
                       const blas_int *lwork, blas_int *info) {
        fortran::dgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
      }

      inline void geev(const char *jobvl, const char *jobvr, const blas_int *n,
                       std::complex<float> *a, const blas_int *lda, std::complex<float> *w,
                       std::complex<float> *vl, const blas_int *ldvl, std::complex<float> *vr,
                       const blas_int *ldvr, std::complex<float> *work, const blas_int *lwork,
                       float *rwork, blas_int *info) {
        fortran::cgeev_(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
      }

      inline void geev(const char *jobvl, const char *jobvr, const blas_int *n,
                       std::complex<double> *a, const blas_int *lda, std::complex<double> *w,
                       std::complex<double> *vl, const blas_int *ldvl, std::complex<double> *vr,
                       const blas_int *ldvr, std::complex<double> *work, const blas_int *lwork,
                       double *rwork, blas_int *info) {
        fortran::zgeev_(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
      }

    }  // namespace native

    template <LapackMatrix Matrix, RealLapackVector Vector>
      requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
    int gesvd(Matrix a, Vector s) {
      using scalar_type = typename Matrix::element_type;
      using real_type = mdspan_concepts::real_element_t<scalar_type>;

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

      if constexpr (RealLapackScalar<scalar_type>) {
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

    template <LapackMatrix Matrix, RealLapackVector Vector, LapackMatrix LeftSingularVectors,
              LapackMatrix RightSingularVectors>
      requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
               mdspan_concepts::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
    int gesvd(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
      using scalar_type = typename Matrix::element_type;
      using real_type = mdspan_concepts::real_element_t<scalar_type>;

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

      if constexpr (RealLapackScalar<scalar_type>) {
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

    template <LapackMatrix Matrix, RealLapackVector Vector>
      requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
    int eigh(char jobz, char uplo, Matrix a, Vector w) {
      using scalar_type = typename Matrix::element_type;
      using real_type = mdspan_concepts::real_element_t<scalar_type>;

      const auto n = a.extent(0);
      cytnx_error_msg(a.extent(1) != n, "[ERROR] LAPACK eigh input must be square.%s", "\n");
      cytnx_error_msg(w.extent(0) < n, "[ERROR] LAPACK eigh eigenvalue output is too small.%s",
                      "\n");

      const blas_int native_n = detail::to_blas_int(n, "n");
      const blas_int lda = std::max<blas_int>(1, native_n);
      const char native_uplo = detail::transpose_uplo(uplo);
      blas_int info = 0;
      blas_int lwork = -1;

      if constexpr (RealLapackScalar<scalar_type>) {
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

    template <LapackMatrix Matrix, ComplexLapackVector Vector>
      requires LapackEigenvalueVector<Matrix, Vector>
    int geev_values(Matrix a, Vector w) {
      using scalar_type = typename Matrix::element_type;
      using real_type = mdspan_concepts::real_element_t<scalar_type>;

      const auto n = a.extent(0);
      cytnx_error_msg(a.extent(1) != n, "[ERROR] LAPACK geev input must be square.%s", "\n");
      cytnx_error_msg(w.extent(0) < n, "[ERROR] LAPACK geev eigenvalue output is too small.%s",
                      "\n");

      const blas_int native_n = detail::to_blas_int(n, "n");
      const blas_int lda = std::max<blas_int>(1, native_n);
      const blas_int one = 1;
      blas_int info = 0;
      blas_int lwork = -1;

      if constexpr (RealLapackScalar<scalar_type>) {
        scalar_type work_query{};
        std::vector<real_type> wr(n);
        std::vector<real_type> wi(n);
        native::geev("N", "N", &native_n, a.data_handle(), &lda, wr.data(), wi.data(), nullptr,
                     &one, nullptr, &one, &work_query, &lwork, &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::geev("N", "N", &native_n, a.data_handle(), &lda, wr.data(), wi.data(), nullptr,
                     &one, nullptr, &one, work.data(), &lwork, &info);
        if (info != 0) return info;
        for (std::size_t i = 0; i < n; ++i) {
          w(i) = typename Vector::element_type{wr[i], wi[i]};
        }
      } else {
        scalar_type work_query{};
        const std::size_t rwork_size = std::max<std::size_t>(1, 2 * n);
        std::vector<real_type> rwork(rwork_size);
        native::geev("N", "N", &native_n, a.data_handle(), &lda, w.data_handle(), nullptr, &one,
                     nullptr, &one, &work_query, &lwork, rwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::geev("N", "N", &native_n, a.data_handle(), &lda, w.data_handle(), nullptr, &one,
                     nullptr, &one, work.data(), &lwork, rwork.data(), &info);
      }

      return info;
    }

  }  // namespace lowlevel

  namespace detail {

    template <class T>
    bool is_finite_scalar(const T &value) {
      if constexpr (ComplexLapackScalar<T>) {
        return std::isfinite(value.real()) && std::isfinite(value.imag());
      } else {
        return std::isfinite(value);
      }
    }

    template <mdspan_concepts::MdspanView View>
    std::size_t nonfinite_count(View view) {
      static_assert(View::rank() == 1 || View::rank() == 2,
                    "LAPACK diagnostics support vector and matrix views");
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

  }  // namespace detail

  /**
   * @brief Compute singular values with checked host LAPACK diagnostics.
   */
  template <LapackMatrix Matrix, RealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void svd_values(Matrix a, Vector s) {
    detail::check_lapack_info("gesvd", lowlevel::gesvd(a, s), a, s);
  }

  /**
   * @brief Compute a thin singular value decomposition with checked host LAPACK diagnostics.
   */
  template <LapackMatrix Matrix, RealLapackVector Vector, LapackMatrix LeftSingularVectors,
            LapackMatrix RightSingularVectors>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
             mdspan_concepts::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
  void svd(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
    detail::check_lapack_info("gesvd", lowlevel::gesvd(a, s, u, vt), a, s, u, vt);
  }

  /**
   * @brief Diagonalize a real symmetric or complex Hermitian matrix with checked host LAPACK
   * diagnostics.
   */
  template <LapackMatrix Matrix, RealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void self_adjoint_eigh(char jobz, char uplo, Matrix a, Vector w) {
    detail::check_lapack_info("eigh", lowlevel::eigh(jobz, uplo, a, w), a, w);
  }

  /**
   * @brief Compute eigenvalues of a general square matrix with checked host LAPACK diagnostics.
   */
  template <LapackMatrix Matrix, ComplexLapackVector Vector>
    requires LapackEigenvalueVector<Matrix, Vector>
  void eig_values(Matrix a, Vector w) {
    detail::check_lapack_info("geev", lowlevel::geev_values(a, w), a, w);
  }

}  // namespace cytnx::lapack

#endif  // BACKEND_TORCH

#endif  // CYTNX_BACKEND_LAPACK_MDSPAN_HPP_
