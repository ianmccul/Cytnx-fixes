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

  void sgesdd_(const char *jobz, const blas_int *m, const blas_int *n, float *a,
               const blas_int *lda, float *s, float *u, const blas_int *ldu, float *vt,
               const blas_int *ldvt, float *work, const blas_int *lwork, blas_int *iwork,
               blas_int *info);
  void dgesdd_(const char *jobz, const blas_int *m, const blas_int *n, double *a,
               const blas_int *lda, double *s, double *u, const blas_int *ldu, double *vt,
               const blas_int *ldvt, double *work, const blas_int *lwork, blas_int *iwork,
               blas_int *info);
  void cgesdd_(const char *jobz, const blas_int *m, const blas_int *n, std::complex<float> *a,
               const blas_int *lda, float *s, std::complex<float> *u, const blas_int *ldu,
               std::complex<float> *vt, const blas_int *ldvt, std::complex<float> *work,
               const blas_int *lwork, float *rwork, blas_int *iwork, blas_int *info);
  void zgesdd_(const char *jobz, const blas_int *m, const blas_int *n, std::complex<double> *a,
               const blas_int *lda, double *s, std::complex<double> *u, const blas_int *ldu,
               std::complex<double> *vt, const blas_int *ldvt, std::complex<double> *work,
               const blas_int *lwork, double *rwork, blas_int *iwork, blas_int *info);

  void sgelsd_(const blas_int *m, const blas_int *n, const blas_int *nrhs, float *a,
               const blas_int *lda, float *b, const blas_int *ldb, float *s, const float *rcond,
               blas_int *rank, float *work, const blas_int *lwork, blas_int *iwork, blas_int *info);
  void dgelsd_(const blas_int *m, const blas_int *n, const blas_int *nrhs, double *a,
               const blas_int *lda, double *b, const blas_int *ldb, double *s, const double *rcond,
               blas_int *rank, double *work, const blas_int *lwork, blas_int *iwork,
               blas_int *info);
  void cgelsd_(const blas_int *m, const blas_int *n, const blas_int *nrhs, std::complex<float> *a,
               const blas_int *lda, std::complex<float> *b, const blas_int *ldb, float *s,
               const float *rcond, blas_int *rank, std::complex<float> *work, const blas_int *lwork,
               float *rwork, blas_int *iwork, blas_int *info);
  void zgelsd_(const blas_int *m, const blas_int *n, const blas_int *nrhs, std::complex<double> *a,
               const blas_int *lda, std::complex<double> *b, const blas_int *ldb, double *s,
               const double *rcond, blas_int *rank, std::complex<double> *work,
               const blas_int *lwork, double *rwork, blas_int *iwork, blas_int *info);

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

  void sstev_(const char *jobz, const blas_int *n, float *d, float *e, float *z,
              const blas_int *ldz, float *work, blas_int *info);
  void dstev_(const char *jobz, const blas_int *n, double *d, double *e, double *z,
              const blas_int *ldz, double *work, blas_int *info);

  void sgetrf_(const blas_int *m, const blas_int *n, float *a, const blas_int *lda, blas_int *ipiv,
               blas_int *info);
  void dgetrf_(const blas_int *m, const blas_int *n, double *a, const blas_int *lda, blas_int *ipiv,
               blas_int *info);
  void cgetrf_(const blas_int *m, const blas_int *n, std::complex<float> *a, const blas_int *lda,
               blas_int *ipiv, blas_int *info);
  void zgetrf_(const blas_int *m, const blas_int *n, std::complex<double> *a, const blas_int *lda,
               blas_int *ipiv, blas_int *info);

  void sgetri_(const blas_int *n, float *a, const blas_int *lda, const blas_int *ipiv, float *work,
               const blas_int *lwork, blas_int *info);
  void dgetri_(const blas_int *n, double *a, const blas_int *lda, const blas_int *ipiv,
               double *work, const blas_int *lwork, blas_int *info);
  void cgetri_(const blas_int *n, std::complex<float> *a, const blas_int *lda, const blas_int *ipiv,
               std::complex<float> *work, const blas_int *lwork, blas_int *info);
  void zgetri_(const blas_int *n, std::complex<double> *a, const blas_int *lda,
               const blas_int *ipiv, std::complex<double> *work, const blas_int *lwork,
               blas_int *info);

  void sgelqf_(const blas_int *m, const blas_int *n, float *a, const blas_int *lda, float *tau,
               float *work, const blas_int *lwork, blas_int *info);
  void dgelqf_(const blas_int *m, const blas_int *n, double *a, const blas_int *lda, double *tau,
               double *work, const blas_int *lwork, blas_int *info);
  void cgelqf_(const blas_int *m, const blas_int *n, std::complex<float> *a, const blas_int *lda,
               std::complex<float> *tau, std::complex<float> *work, const blas_int *lwork,
               blas_int *info);
  void zgelqf_(const blas_int *m, const blas_int *n, std::complex<double> *a, const blas_int *lda,
               std::complex<double> *tau, std::complex<double> *work, const blas_int *lwork,
               blas_int *info);
  void sorglq_(const blas_int *m, const blas_int *n, const blas_int *k, float *a,
               const blas_int *lda, const float *tau, float *work, const blas_int *lwork,
               blas_int *info);
  void dorglq_(const blas_int *m, const blas_int *n, const blas_int *k, double *a,
               const blas_int *lda, const double *tau, double *work, const blas_int *lwork,
               blas_int *info);
  void cunglq_(const blas_int *m, const blas_int *n, const blas_int *k, std::complex<float> *a,
               const blas_int *lda, const std::complex<float> *tau, std::complex<float> *work,
               const blas_int *lwork, blas_int *info);
  void zunglq_(const blas_int *m, const blas_int *n, const blas_int *k, std::complex<double> *a,
               const blas_int *lda, const std::complex<double> *tau, std::complex<double> *work,
               const blas_int *lwork, blas_int *info);

  void sgeqrf_(const blas_int *m, const blas_int *n, float *a, const blas_int *lda, float *tau,
               float *work, const blas_int *lwork, blas_int *info);
  void dgeqrf_(const blas_int *m, const blas_int *n, double *a, const blas_int *lda, double *tau,
               double *work, const blas_int *lwork, blas_int *info);
  void cgeqrf_(const blas_int *m, const blas_int *n, std::complex<float> *a, const blas_int *lda,
               std::complex<float> *tau, std::complex<float> *work, const blas_int *lwork,
               blas_int *info);
  void zgeqrf_(const blas_int *m, const blas_int *n, std::complex<double> *a, const blas_int *lda,
               std::complex<double> *tau, std::complex<double> *work, const blas_int *lwork,
               blas_int *info);
  void sorgqr_(const blas_int *m, const blas_int *n, const blas_int *k, float *a,
               const blas_int *lda, const float *tau, float *work, const blas_int *lwork,
               blas_int *info);
  void dorgqr_(const blas_int *m, const blas_int *n, const blas_int *k, double *a,
               const blas_int *lda, const double *tau, double *work, const blas_int *lwork,
               blas_int *info);
  void cungqr_(const blas_int *m, const blas_int *n, const blas_int *k, std::complex<float> *a,
               const blas_int *lda, const std::complex<float> *tau, std::complex<float> *work,
               const blas_int *lwork, blas_int *info);
  void zungqr_(const blas_int *m, const blas_int *n, const blas_int *k, std::complex<double> *a,
               const blas_int *lda, const std::complex<double> *tau, std::complex<double> *work,
               const blas_int *lwork, blas_int *info);
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

    template <class T>
    T conjugate_if_complex(const T &value) {
      if constexpr (std::is_same_v<std::remove_cv_t<T>, std::complex<float>> ||
                    std::is_same_v<std::remove_cv_t<T>, std::complex<double>>) {
        return std::conj(value);
      } else {
        return value;
      }
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

  template <class View>
  concept MutableLapackVector = LapackVector<View> && mdspan_concepts::MutableView<View>;

  template <class View>
  concept MutableRealLapackVector = RealLapackVector<View> && mdspan_concepts::MutableView<View>;

  template <class View>
  concept MutableComplexLapackVector =
    ComplexLapackVector<View> && mdspan_concepts::MutableView<View>;

  template <class Matrix, class Vector>
  concept LapackEigenvalueVector =
    ComplexLapackVector<Vector> &&
    ((RealLapackScalar<typename Matrix::element_type> &&
      std::same_as<mdspan_concepts::element_value_t<Vector>,
                   std::complex<mdspan_concepts::element_value_t<Matrix>>>) ||
     (ComplexLapackScalar<typename Matrix::element_type> &&
      std::same_as<mdspan_concepts::element_value_t<Matrix>,
                   mdspan_concepts::element_value_t<Vector>>));

  template <class View>
  concept LapackMatrix =
    mdspan_concepts::LayoutRightMatrix<View> && mdspan_concepts::HostAccessible<View> &&
    LapackScalar<typename View::element_type>;

  template <class View>
  concept RealLapackMatrix = LapackMatrix<View> && RealLapackScalar<typename View::element_type>;

  template <class View>
  concept ComplexLapackMatrix =
    LapackMatrix<View> && ComplexLapackScalar<typename View::element_type>;

  template <class View>
  concept MutableLapackMatrix = LapackMatrix<View> && mdspan_concepts::MutableView<View>;

  template <class View>
  concept MutableRealLapackMatrix = RealLapackMatrix<View> && mdspan_concepts::MutableView<View>;

  template <class View>
  concept MutableComplexLapackMatrix =
    ComplexLapackMatrix<View> && mdspan_concepts::MutableView<View>;

  template <class Matrix, class Eigenvectors>
  concept LapackEigenvectorMatrix =
    MutableComplexLapackMatrix<Eigenvectors> &&
    ((RealLapackScalar<typename Matrix::element_type> &&
      std::same_as<mdspan_concepts::element_value_t<Eigenvectors>,
                   std::complex<mdspan_concepts::element_value_t<Matrix>>>) ||
     (ComplexLapackScalar<typename Matrix::element_type> &&
      std::same_as<mdspan_concepts::element_value_t<Matrix>,
                   mdspan_concepts::element_value_t<Eigenvectors>>));

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

      inline void gesdd(const char *jobz, const blas_int *m, const blas_int *n, float *a,
                        const blas_int *lda, float *s, float *u, const blas_int *ldu, float *vt,
                        const blas_int *ldvt, float *work, const blas_int *lwork, blas_int *iwork,
                        blas_int *info) {
        fortran::sgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
      }

      inline void gesdd(const char *jobz, const blas_int *m, const blas_int *n, double *a,
                        const blas_int *lda, double *s, double *u, const blas_int *ldu, double *vt,
                        const blas_int *ldvt, double *work, const blas_int *lwork, blas_int *iwork,
                        blas_int *info) {
        fortran::dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
      }

      inline void gesdd(const char *jobz, const blas_int *m, const blas_int *n,
                        std::complex<float> *a, const blas_int *lda, float *s,
                        std::complex<float> *u, const blas_int *ldu, std::complex<float> *vt,
                        const blas_int *ldvt, std::complex<float> *work, const blas_int *lwork,
                        float *rwork, blas_int *iwork, blas_int *info) {
        fortran::cgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
      }

      inline void gesdd(const char *jobz, const blas_int *m, const blas_int *n,
                        std::complex<double> *a, const blas_int *lda, double *s,
                        std::complex<double> *u, const blas_int *ldu, std::complex<double> *vt,
                        const blas_int *ldvt, std::complex<double> *work, const blas_int *lwork,
                        double *rwork, blas_int *iwork, blas_int *info) {
        fortran::zgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
      }

      inline void gelsd(const blas_int *m, const blas_int *n, const blas_int *nrhs, float *a,
                        const blas_int *lda, float *b, const blas_int *ldb, float *s,
                        const float *rcond, blas_int *rank, float *work, const blas_int *lwork,
                        blas_int *iwork, blas_int *info) {
        fortran::sgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info);
      }

      inline void gelsd(const blas_int *m, const blas_int *n, const blas_int *nrhs, double *a,
                        const blas_int *lda, double *b, const blas_int *ldb, double *s,
                        const double *rcond, blas_int *rank, double *work, const blas_int *lwork,
                        blas_int *iwork, blas_int *info) {
        fortran::dgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info);
      }

      inline void gelsd(const blas_int *m, const blas_int *n, const blas_int *nrhs,
                        std::complex<float> *a, const blas_int *lda, std::complex<float> *b,
                        const blas_int *ldb, float *s, const float *rcond, blas_int *rank,
                        std::complex<float> *work, const blas_int *lwork, float *rwork,
                        blas_int *iwork, blas_int *info) {
        fortran::cgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, rwork, iwork,
                         info);
      }

      inline void gelsd(const blas_int *m, const blas_int *n, const blas_int *nrhs,
                        std::complex<double> *a, const blas_int *lda, std::complex<double> *b,
                        const blas_int *ldb, double *s, const double *rcond, blas_int *rank,
                        std::complex<double> *work, const blas_int *lwork, double *rwork,
                        blas_int *iwork, blas_int *info) {
        fortran::zgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, rwork, iwork,
                         info);
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

      inline void stev(const char *jobz, const blas_int *n, float *d, float *e, float *z,
                       const blas_int *ldz, float *work, blas_int *info) {
        fortran::sstev_(jobz, n, d, e, z, ldz, work, info);
      }

      inline void stev(const char *jobz, const blas_int *n, double *d, double *e, double *z,
                       const blas_int *ldz, double *work, blas_int *info) {
        fortran::dstev_(jobz, n, d, e, z, ldz, work, info);
      }

      inline void getrf(const blas_int *m, const blas_int *n, float *a, const blas_int *lda,
                        blas_int *ipiv, blas_int *info) {
        fortran::sgetrf_(m, n, a, lda, ipiv, info);
      }

      inline void getrf(const blas_int *m, const blas_int *n, double *a, const blas_int *lda,
                        blas_int *ipiv, blas_int *info) {
        fortran::dgetrf_(m, n, a, lda, ipiv, info);
      }

      inline void getrf(const blas_int *m, const blas_int *n, std::complex<float> *a,
                        const blas_int *lda, blas_int *ipiv, blas_int *info) {
        fortran::cgetrf_(m, n, a, lda, ipiv, info);
      }

      inline void getrf(const blas_int *m, const blas_int *n, std::complex<double> *a,
                        const blas_int *lda, blas_int *ipiv, blas_int *info) {
        fortran::zgetrf_(m, n, a, lda, ipiv, info);
      }

      inline void getri(const blas_int *n, float *a, const blas_int *lda, const blas_int *ipiv,
                        float *work, const blas_int *lwork, blas_int *info) {
        fortran::sgetri_(n, a, lda, ipiv, work, lwork, info);
      }

      inline void getri(const blas_int *n, double *a, const blas_int *lda, const blas_int *ipiv,
                        double *work, const blas_int *lwork, blas_int *info) {
        fortran::dgetri_(n, a, lda, ipiv, work, lwork, info);
      }

      inline void getri(const blas_int *n, std::complex<float> *a, const blas_int *lda,
                        const blas_int *ipiv, std::complex<float> *work, const blas_int *lwork,
                        blas_int *info) {
        fortran::cgetri_(n, a, lda, ipiv, work, lwork, info);
      }

      inline void getri(const blas_int *n, std::complex<double> *a, const blas_int *lda,
                        const blas_int *ipiv, std::complex<double> *work, const blas_int *lwork,
                        blas_int *info) {
        fortran::zgetri_(n, a, lda, ipiv, work, lwork, info);
      }

      inline void gelqf(const blas_int *m, const blas_int *n, float *a, const blas_int *lda,
                        float *tau, float *work, const blas_int *lwork, blas_int *info) {
        fortran::sgelqf_(m, n, a, lda, tau, work, lwork, info);
      }

      inline void gelqf(const blas_int *m, const blas_int *n, double *a, const blas_int *lda,
                        double *tau, double *work, const blas_int *lwork, blas_int *info) {
        fortran::dgelqf_(m, n, a, lda, tau, work, lwork, info);
      }

      inline void gelqf(const blas_int *m, const blas_int *n, std::complex<float> *a,
                        const blas_int *lda, std::complex<float> *tau, std::complex<float> *work,
                        const blas_int *lwork, blas_int *info) {
        fortran::cgelqf_(m, n, a, lda, tau, work, lwork, info);
      }

      inline void gelqf(const blas_int *m, const blas_int *n, std::complex<double> *a,
                        const blas_int *lda, std::complex<double> *tau, std::complex<double> *work,
                        const blas_int *lwork, blas_int *info) {
        fortran::zgelqf_(m, n, a, lda, tau, work, lwork, info);
      }

      inline void orglq(const blas_int *m, const blas_int *n, const blas_int *k, float *a,
                        const blas_int *lda, const float *tau, float *work, const blas_int *lwork,
                        blas_int *info) {
        fortran::sorglq_(m, n, k, a, lda, tau, work, lwork, info);
      }

      inline void orglq(const blas_int *m, const blas_int *n, const blas_int *k, double *a,
                        const blas_int *lda, const double *tau, double *work, const blas_int *lwork,
                        blas_int *info) {
        fortran::dorglq_(m, n, k, a, lda, tau, work, lwork, info);
      }

      inline void orglq(const blas_int *m, const blas_int *n, const blas_int *k,
                        std::complex<float> *a, const blas_int *lda, const std::complex<float> *tau,
                        std::complex<float> *work, const blas_int *lwork, blas_int *info) {
        fortran::cunglq_(m, n, k, a, lda, tau, work, lwork, info);
      }

      inline void orglq(const blas_int *m, const blas_int *n, const blas_int *k,
                        std::complex<double> *a, const blas_int *lda,
                        const std::complex<double> *tau, std::complex<double> *work,
                        const blas_int *lwork, blas_int *info) {
        fortran::zunglq_(m, n, k, a, lda, tau, work, lwork, info);
      }

      inline void geqrf(const blas_int *m, const blas_int *n, float *a, const blas_int *lda,
                        float *tau, float *work, const blas_int *lwork, blas_int *info) {
        fortran::sgeqrf_(m, n, a, lda, tau, work, lwork, info);
      }

      inline void geqrf(const blas_int *m, const blas_int *n, double *a, const blas_int *lda,
                        double *tau, double *work, const blas_int *lwork, blas_int *info) {
        fortran::dgeqrf_(m, n, a, lda, tau, work, lwork, info);
      }

      inline void geqrf(const blas_int *m, const blas_int *n, std::complex<float> *a,
                        const blas_int *lda, std::complex<float> *tau, std::complex<float> *work,
                        const blas_int *lwork, blas_int *info) {
        fortran::cgeqrf_(m, n, a, lda, tau, work, lwork, info);
      }

      inline void geqrf(const blas_int *m, const blas_int *n, std::complex<double> *a,
                        const blas_int *lda, std::complex<double> *tau, std::complex<double> *work,
                        const blas_int *lwork, blas_int *info) {
        fortran::zgeqrf_(m, n, a, lda, tau, work, lwork, info);
      }

      inline void orgqr(const blas_int *m, const blas_int *n, const blas_int *k, float *a,
                        const blas_int *lda, const float *tau, float *work, const blas_int *lwork,
                        blas_int *info) {
        fortran::sorgqr_(m, n, k, a, lda, tau, work, lwork, info);
      }

      inline void orgqr(const blas_int *m, const blas_int *n, const blas_int *k, double *a,
                        const blas_int *lda, const double *tau, double *work, const blas_int *lwork,
                        blas_int *info) {
        fortran::dorgqr_(m, n, k, a, lda, tau, work, lwork, info);
      }

      inline void orgqr(const blas_int *m, const blas_int *n, const blas_int *k,
                        std::complex<float> *a, const blas_int *lda, const std::complex<float> *tau,
                        std::complex<float> *work, const blas_int *lwork, blas_int *info) {
        fortran::cungqr_(m, n, k, a, lda, tau, work, lwork, info);
      }

      inline void orgqr(const blas_int *m, const blas_int *n, const blas_int *k,
                        std::complex<double> *a, const blas_int *lda,
                        const std::complex<double> *tau, std::complex<double> *work,
                        const blas_int *lwork, blas_int *info) {
        fortran::zungqr_(m, n, k, a, lda, tau, work, lwork, info);
      }

    }  // namespace native

    template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector>
      requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
    int gesvd(Matrix a, Vector s) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;
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

    template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector,
              MutableLapackMatrix LeftSingularVectors, MutableLapackMatrix RightSingularVectors>
      requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
               mdspan_concepts::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
    int gesvd(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;
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

    template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector>
      requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
    int gesdd(Matrix a, Vector s) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;
      using real_type = mdspan_concepts::real_element_t<scalar_type>;

      const auto rows = a.extent(0);
      const auto cols = a.extent(1);
      const auto min_dim = std::min(rows, cols);
      cytnx_error_msg(s.extent(0) < min_dim,
                      "[ERROR] LAPACK gesdd singular-value output is too small.%s", "\n");

      const blas_int native_m = detail::to_blas_int(cols, "cols");
      const blas_int native_n = detail::to_blas_int(rows, "rows");
      const blas_int lda = std::max<blas_int>(1, native_m);
      const blas_int one = 1;
      blas_int info = 0;
      blas_int lwork = -1;
      std::vector<blas_int> iwork(std::max<std::size_t>(1, 8 * min_dim));

      if constexpr (RealLapackScalar<scalar_type>) {
        scalar_type work_query{};
        native::gesdd("N", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(), nullptr,
                      &one, nullptr, &one, &work_query, &lwork, iwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gesdd("N", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(), nullptr,
                      &one, nullptr, &one, work.data(), &lwork, iwork.data(), &info);
      } else {
        scalar_type work_query{};
        std::vector<real_type> rwork(std::max<std::size_t>(1, 5 * min_dim));
        native::gesdd("N", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(), nullptr,
                      &one, nullptr, &one, &work_query, &lwork, rwork.data(), iwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gesdd("N", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(), nullptr,
                      &one, nullptr, &one, work.data(), &lwork, rwork.data(), iwork.data(), &info);
      }

      return info;
    }

    template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector,
              MutableLapackMatrix LeftSingularVectors, MutableLapackMatrix RightSingularVectors>
      requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
               mdspan_concepts::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
    int gesdd(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;
      using real_type = mdspan_concepts::real_element_t<scalar_type>;

      const auto rows = a.extent(0);
      const auto cols = a.extent(1);
      const auto min_dim = std::min(rows, cols);
      const auto max_dim = std::max(rows, cols);
      cytnx_error_msg(s.extent(0) < min_dim,
                      "[ERROR] LAPACK gesdd singular-value output is too small.%s", "\n");
      cytnx_error_msg(u.extent(0) < rows || u.extent(1) < min_dim,
                      "[ERROR] LAPACK gesdd U output has incompatible shape.%s", "\n");
      cytnx_error_msg(vt.extent(0) < min_dim || vt.extent(1) < cols,
                      "[ERROR] LAPACK gesdd VT output has incompatible shape.%s", "\n");

      const blas_int native_m = detail::to_blas_int(cols, "cols");
      const blas_int native_n = detail::to_blas_int(rows, "rows");
      const blas_int lda = std::max<blas_int>(1, native_m);
      const blas_int native_ldu = std::max<blas_int>(1, native_m);
      const blas_int native_ldvt = std::max<blas_int>(1, detail::to_blas_int(min_dim, "min_dim"));
      blas_int info = 0;
      blas_int lwork = -1;
      std::vector<blas_int> iwork(std::max<std::size_t>(1, 8 * min_dim));

      if constexpr (RealLapackScalar<scalar_type>) {
        scalar_type work_query{};
        native::gesdd("S", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      vt.data_handle(), &native_ldu, u.data_handle(), &native_ldvt, &work_query,
                      &lwork, iwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gesdd("S", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      vt.data_handle(), &native_ldu, u.data_handle(), &native_ldvt, work.data(),
                      &lwork, iwork.data(), &info);
      } else {
        scalar_type work_query{};
        const std::size_t rwork_size =
          std::max<std::size_t>(1, 5 * min_dim * min_dim + 7 * min_dim + 2 * max_dim * min_dim);
        std::vector<real_type> rwork(rwork_size);
        native::gesdd("S", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      vt.data_handle(), &native_ldu, u.data_handle(), &native_ldvt, &work_query,
                      &lwork, rwork.data(), iwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gesdd("S", &native_m, &native_n, a.data_handle(), &lda, s.data_handle(),
                      vt.data_handle(), &native_ldu, u.data_handle(), &native_ldvt, work.data(),
                      &lwork, rwork.data(), iwork.data(), &info);
      }

      return info;
    }

    template <LapackMatrix Matrix, MutableLapackMatrix RightHandSide,
              MutableRealLapackVector Vector>
      requires mdspan_concepts::SameElementType<Matrix, RightHandSide> &&
               mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
    int gelsd(Matrix a, RightHandSide b, Vector s, blas_int &rank,
              mdspan_concepts::real_element_t<mdspan_concepts::element_value_t<Matrix>> rcond) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;
      using real_type = mdspan_concepts::real_element_t<scalar_type>;

      const auto rows = a.extent(0);
      const auto cols = a.extent(1);
      const auto rhs_count = b.extent(1);
      const auto min_dim = std::min(rows, cols);
      const auto max_dim = std::max(rows, cols);
      cytnx_error_msg(b.extent(0) < max_dim,
                      "[ERROR] LAPACK gelsd right-hand side matrix has too few rows.%s", "\n");
      cytnx_error_msg(s.extent(0) < min_dim,
                      "[ERROR] LAPACK gelsd singular-value output is too small.%s", "\n");

      const blas_int native_m = detail::to_blas_int(rows, "rows");
      const blas_int native_n = detail::to_blas_int(cols, "cols");
      const blas_int native_nrhs = detail::to_blas_int(rhs_count, "rhs_count");
      const blas_int lda = std::max<blas_int>(1, native_m);
      const blas_int ldb = std::max<blas_int>(1, detail::to_blas_int(max_dim, "max_dim"));
      blas_int info = 0;
      blas_int lwork = -1;

      std::vector<scalar_type> a_column_major(
        std::max<std::size_t>(1, static_cast<std::size_t>(lda) * cols));
      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
          a_column_major[i + j * static_cast<std::size_t>(lda)] = a(i, j);
        }
      }

      std::vector<scalar_type> b_column_major(
        std::max<std::size_t>(1, static_cast<std::size_t>(ldb) * rhs_count));
      for (std::size_t i = 0; i < max_dim; ++i) {
        for (std::size_t j = 0; j < rhs_count; ++j) {
          b_column_major[i + j * static_cast<std::size_t>(ldb)] = b(i, j);
        }
      }

      if constexpr (RealLapackScalar<scalar_type>) {
        scalar_type work_query{};
        std::vector<blas_int> iwork(1);
        native::gelsd(&native_m, &native_n, &native_nrhs, a_column_major.data(), &lda,
                      b_column_major.data(), &ldb, s.data_handle(), &rcond, &rank, &work_query,
                      &lwork, iwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
        iwork.assign(std::max<blas_int>(1, iwork[0]), blas_int{});
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gelsd(&native_m, &native_n, &native_nrhs, a_column_major.data(), &lda,
                      b_column_major.data(), &ldb, s.data_handle(), &rcond, &rank, work.data(),
                      &lwork, iwork.data(), &info);
      } else {
        scalar_type work_query{};
        std::vector<real_type> rwork(1);
        std::vector<blas_int> iwork(1);
        native::gelsd(&native_m, &native_n, &native_nrhs, a_column_major.data(), &lda,
                      b_column_major.data(), &ldb, s.data_handle(), &rcond, &rank, &work_query,
                      &lwork, rwork.data(), iwork.data(), &info);
        if (info != 0) return info;
        lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
        rwork.assign(std::max<blas_int>(1, static_cast<blas_int>(rwork[0])), real_type{});
        iwork.assign(std::max<blas_int>(1, iwork[0]), blas_int{});
        std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
        native::gelsd(&native_m, &native_n, &native_nrhs, a_column_major.data(), &lda,
                      b_column_major.data(), &ldb, s.data_handle(), &rcond, &rank, work.data(),
                      &lwork, rwork.data(), iwork.data(), &info);
      }

      if (info != 0) return info;
      for (std::size_t i = 0; i < max_dim; ++i) {
        for (std::size_t j = 0; j < rhs_count; ++j) {
          b(i, j) = b_column_major[i + j * static_cast<std::size_t>(ldb)];
        }
      }
      return info;
    }

    template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector>
      requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
    int eigh(char jobz, char uplo, Matrix a, Vector w) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;
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

    template <MutableLapackMatrix Matrix, MutableComplexLapackVector Vector>
      requires LapackEigenvalueVector<Matrix, Vector>
    int geev_values(Matrix a, Vector w) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;
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
          w(i) = mdspan_concepts::element_value_t<Vector>{wr[i], wi[i]};
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

    template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector,
              MutableLapackMatrix Eigenvectors>
      requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
               mdspan_concepts::SameElementType<Matrix, Eigenvectors>
    int eigh_vectors(char uplo, Matrix a, Vector w, Eigenvectors vectors) {
      const auto n = a.extent(0);
      cytnx_error_msg(vectors.extent(0) < n || vectors.extent(1) < n,
                      "[ERROR] LAPACK eigh eigenvector output has incompatible shape.%s", "\n");

      const int info = eigh('V', uplo, a, w);
      if (info != 0) return info;

      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          vectors(i, j) = detail::conjugate_if_complex(a(i, j));
        }
      }
      return info;
    }

    template <LapackMatrix Matrix, MutableComplexLapackVector Vector,
              MutableComplexLapackMatrix Eigenvectors>
      requires LapackEigenvalueVector<Matrix, Vector> &&
               LapackEigenvectorMatrix<Matrix, Eigenvectors>
    int geev_right_vectors(Matrix a, Vector w, Eigenvectors vectors) {
      using complex_type = mdspan_concepts::element_value_t<Vector>;
      using real_type = mdspan_concepts::real_element_t<complex_type>;

      const auto n = a.extent(0);
      cytnx_error_msg(a.extent(1) != n, "[ERROR] LAPACK geev input must be square.%s", "\n");
      cytnx_error_msg(w.extent(0) < n, "[ERROR] LAPACK geev eigenvalue output is too small.%s",
                      "\n");
      cytnx_error_msg(vectors.extent(0) < n || vectors.extent(1) < n,
                      "[ERROR] LAPACK geev eigenvector output has incompatible shape.%s", "\n");

      const blas_int native_n = detail::to_blas_int(n, "n");
      const blas_int lda = std::max<blas_int>(1, native_n);
      const blas_int one = 1;
      blas_int info = 0;
      blas_int lwork = -1;

      std::vector<complex_type> buffer(n * n);
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          buffer[i + j * n] = complex_type{a(i, j)};
        }
      }

      complex_type work_query{};
      std::vector<real_type> rwork(std::max<std::size_t>(1, 2 * n));
      native::geev("N", "V", &native_n, buffer.data(), &lda, w.data_handle(), nullptr, &one,
                   vectors.data_handle(), &lda, &work_query, &lwork, rwork.data(), &info);
      if (info != 0) return info;
      lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
      std::vector<complex_type> work(static_cast<std::size_t>(lwork));
      native::geev("N", "V", &native_n, buffer.data(), &lda, w.data_handle(), nullptr, &one,
                   vectors.data_handle(), &lda, work.data(), &lwork, rwork.data(), &info);
      if (info != 0) return info;

      // LAPACK writes right eigenvectors as columns in column-major storage. The same memory viewed
      // as row-major has eigenvectors as rows: vectors(i, j) is component j of eigenvector i.
      return info;
    }

    template <MutableRealLapackVector Diagonal, MutableRealLapackVector OffDiagonal>
      requires mdspan_concepts::SameElementType<Diagonal, OffDiagonal>
    int stev_values(Diagonal diagonal, OffDiagonal offdiagonal) {
      using scalar_type = mdspan_concepts::element_value_t<Diagonal>;

      const auto n = diagonal.extent(0);
      const auto min_offdiag = n > 0 ? n - 1 : 0;
      cytnx_error_msg(offdiagonal.extent(0) < min_offdiag,
                      "[ERROR] LAPACK stev off-diagonal input is too small.%s", "\n");

      const blas_int native_n = detail::to_blas_int(n, "n");
      const blas_int ldz = 1;
      blas_int info = 0;
      const std::size_t work_size = std::max<std::size_t>(1, n > 1 ? 2 * n - 2 : 1);
      std::vector<scalar_type> work(work_size);

      native::stev("N", &native_n, diagonal.data_handle(), offdiagonal.data_handle(), nullptr, &ldz,
                   work.data(), &info);
      return info;
    }

    template <MutableLapackMatrix Matrix>
    int getri_inplace(Matrix a) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;

      const auto n = a.extent(0);
      cytnx_error_msg(a.extent(1) != n, "[ERROR] LAPACK getri input must be square.%s", "\n");

      const blas_int native_n = detail::to_blas_int(n, "n");
      const blas_int lda = std::max<blas_int>(1, native_n);
      blas_int info = 0;
      std::vector<blas_int> pivots(n);

      native::getrf(&native_n, &native_n, a.data_handle(), &lda, pivots.data(), &info);
      if (info != 0) return info;

      blas_int lwork = -1;
      scalar_type work_query{};
      native::getri(&native_n, a.data_handle(), &lda, pivots.data(), &work_query, &lwork, &info);
      if (info != 0) return info;
      if constexpr (RealLapackScalar<scalar_type>) {
        lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
      } else {
        lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
      }
      std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
      native::getri(&native_n, a.data_handle(), &lda, pivots.data(), work.data(), &lwork, &info);
      return info;
    }

    template <LapackMatrix Matrix, MutableLapackMatrix QMatrix, MutableLapackMatrix RMatrix>
      requires mdspan_concepts::SameElementType<Matrix, QMatrix, RMatrix>
    int qr(Matrix a, QMatrix q, RMatrix r) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;

      const auto rows = a.extent(0);
      const auto cols = a.extent(1);
      const auto k = std::min(rows, cols);
      cytnx_error_msg(q.extent(0) < rows || q.extent(1) < k,
                      "[ERROR] LAPACK qr Q output has incompatible shape.%s", "\n");
      cytnx_error_msg(r.extent(0) < k || r.extent(1) < cols,
                      "[ERROR] LAPACK qr R output has incompatible shape.%s", "\n");

      const blas_int native_rows = detail::to_blas_int(cols, "cols");
      const blas_int native_cols = detail::to_blas_int(rows, "rows");
      const blas_int native_k = detail::to_blas_int(k, "k");
      const blas_int lda = std::max<blas_int>(1, native_rows);
      blas_int info = 0;
      blas_int lwork = -1;
      std::vector<scalar_type> buffer(rows * cols);
      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) buffer[i * cols + j] = a(i, j);
      }
      std::vector<scalar_type> tau(k);

      // layout_right stores logical A as the column-major matrix A^T for native LAPACK.
      // Therefore logical QR(A) is obtained from LQ(A^T): A^T = L_t Q_t, so
      // A = Q_t^T L_t^T.
      scalar_type work_query{};
      native::gelqf(&native_rows, &native_cols, buffer.data(), &lda, tau.data(), &work_query,
                    &lwork, &info);
      if (info != 0) return info;
      lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
      std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
      native::gelqf(&native_rows, &native_cols, buffer.data(), &lda, tau.data(), work.data(),
                    &lwork, &info);
      if (info != 0) return info;

      for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
          r(i, j) = j >= i ? buffer[j + i * static_cast<std::size_t>(lda)] : scalar_type{};
        }
      }

      lwork = -1;
      native::orglq(&native_k, &native_cols, &native_k, buffer.data(), &lda, tau.data(),
                    &work_query, &lwork, &info);
      if (info != 0) return info;
      lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
      work.assign(static_cast<std::size_t>(lwork), scalar_type{});
      native::orglq(&native_k, &native_cols, &native_k, buffer.data(), &lda, tau.data(),
                    work.data(), &lwork, &info);
      if (info != 0) return info;

      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
          q(i, j) = buffer[j + i * static_cast<std::size_t>(lda)];
        }
      }

      return info;
    }

    template <LapackMatrix Matrix, MutableLapackMatrix LMatrix, MutableLapackMatrix QMatrix>
      requires mdspan_concepts::SameElementType<Matrix, LMatrix, QMatrix>
    int lq(Matrix a, LMatrix l, QMatrix q) {
      using scalar_type = mdspan_concepts::element_value_t<Matrix>;

      const auto rows = a.extent(0);
      const auto cols = a.extent(1);
      const auto k = std::min(rows, cols);
      cytnx_error_msg(l.extent(0) < rows || l.extent(1) < k,
                      "[ERROR] LAPACK lq L output has incompatible shape.%s", "\n");
      cytnx_error_msg(q.extent(0) < k || q.extent(1) < cols,
                      "[ERROR] LAPACK lq Q output has incompatible shape.%s", "\n");

      const blas_int native_rows = detail::to_blas_int(cols, "cols");
      const blas_int native_cols = detail::to_blas_int(rows, "rows");
      const blas_int native_k = detail::to_blas_int(k, "k");
      const blas_int lda = std::max<blas_int>(1, native_rows);
      blas_int info = 0;
      blas_int lwork = -1;
      std::vector<scalar_type> buffer(rows * cols);
      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) buffer[i * cols + j] = a(i, j);
      }
      std::vector<scalar_type> tau(k);

      // layout_right stores logical A as the column-major matrix A^T for native LAPACK.
      // Therefore logical LQ(A) is obtained from QR(A^T): A^T = Q_t R_t, so
      // A = R_t^T Q_t^T.
      scalar_type work_query{};
      native::geqrf(&native_rows, &native_cols, buffer.data(), &lda, tau.data(), &work_query,
                    &lwork, &info);
      if (info != 0) return info;
      lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
      std::vector<scalar_type> work(static_cast<std::size_t>(lwork));
      native::geqrf(&native_rows, &native_cols, buffer.data(), &lda, tau.data(), work.data(),
                    &lwork, &info);
      if (info != 0) return info;

      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
          l(i, j) = i >= j ? buffer[j + i * static_cast<std::size_t>(lda)] : scalar_type{};
        }
      }

      lwork = -1;
      native::orgqr(&native_rows, &native_k, &native_k, buffer.data(), &lda, tau.data(),
                    &work_query, &lwork, &info);
      if (info != 0) return info;
      lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
      work.assign(static_cast<std::size_t>(lwork), scalar_type{});
      native::orgqr(&native_rows, &native_k, &native_k, buffer.data(), &lda, tau.data(),
                    work.data(), &lwork, &info);
      if (info != 0) return info;

      for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
          q(i, j) = buffer[j + i * static_cast<std::size_t>(lda)];
        }
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
  template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void svd_values(Matrix a, Vector s) {
    detail::check_lapack_info("gesvd", lowlevel::gesvd(a, s), a, s);
  }

  /**
   * @brief Compute a thin singular value decomposition with checked host LAPACK diagnostics.
   */
  template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector,
            MutableLapackMatrix LeftSingularVectors, MutableLapackMatrix RightSingularVectors>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
             mdspan_concepts::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
  void svd(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
    detail::check_lapack_info("gesvd", lowlevel::gesvd(a, s, u, vt), a, s, u, vt);
  }

  /**
   * @brief Compute singular values with the divide-and-conquer SVD driver.
   */
  template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void svd_divide_conquer_values(Matrix a, Vector s) {
    detail::check_lapack_info("gesdd", lowlevel::gesdd(a, s), a, s);
  }

  /**
   * @brief Compute a thin singular value decomposition with the divide-and-conquer SVD driver.
   */
  template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector,
            MutableLapackMatrix LeftSingularVectors, MutableLapackMatrix RightSingularVectors>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
             mdspan_concepts::SameElementType<Matrix, LeftSingularVectors, RightSingularVectors>
  void svd_divide_conquer(Matrix a, Vector s, LeftSingularVectors u, RightSingularVectors vt) {
    detail::check_lapack_info("gesdd", lowlevel::gesdd(a, s, u, vt), a, s, u, vt);
  }

  /**
   * @brief Solve a least-squares problem with the divide-and-conquer SVD driver.
   *
   * `b` must have shape at least `(max(a.extent(0), a.extent(1)), nrhs)`. On return, the first
   * `a.extent(1)` rows of `b` contain the solution.
   */
  template <LapackMatrix Matrix, MutableLapackMatrix RightHandSide, MutableRealLapackVector Vector,
            class Rcond>
    requires mdspan_concepts::SameElementType<Matrix, RightHandSide> &&
             mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
             std::convertible_to<
               Rcond, mdspan_concepts::real_element_t<mdspan_concepts::element_value_t<Matrix>>>
  void least_squares(Matrix a, RightHandSide b, Vector s, blas_int &rank, Rcond rcond) {
    using real_type = mdspan_concepts::real_element_t<mdspan_concepts::element_value_t<Matrix>>;
    detail::check_lapack_info(
      "gelsd", lowlevel::gelsd(a, b, s, rank, static_cast<real_type>(rcond)), a, b, s);
  }

  /**
   * @brief Diagonalize a real symmetric or complex Hermitian matrix with checked host LAPACK
   * diagnostics.
   */
  template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>>
  void self_adjoint_eigh(char jobz, char uplo, Matrix a, Vector w) {
    detail::check_lapack_info("eigh", lowlevel::eigh(jobz, uplo, a, w), a, w);
  }

  /**
   * @brief Diagonalize a real symmetric or complex Hermitian matrix and write eigenvectors as rows.
   */
  template <MutableLapackMatrix Matrix, MutableRealLapackVector Vector,
            MutableLapackMatrix Eigenvectors>
    requires mdspan_concepts::SameElementType<Vector, mdspan_concepts::RealElementOf<Matrix>> &&
             mdspan_concepts::SameElementType<Matrix, Eigenvectors>
  void self_adjoint_eigh_vectors(char uplo, Matrix a, Vector w, Eigenvectors vectors) {
    detail::check_lapack_info("eigh", lowlevel::eigh_vectors(uplo, a, w, vectors), a, w, vectors);
  }

  /**
   * @brief Compute eigenvalues of a general square matrix with checked host LAPACK diagnostics.
   */
  template <MutableLapackMatrix Matrix, MutableComplexLapackVector Vector>
    requires LapackEigenvalueVector<Matrix, Vector>
  void eig_values(Matrix a, Vector w) {
    detail::check_lapack_info("geev", lowlevel::geev_values(a, w), a, w);
  }

  /**
   * @brief Compute eigenvalues and right eigenvectors of a general square matrix.
   *
   * Eigenvectors are written as rows: `vectors(i, j)` is component `j` of eigenvector `i`.
   */
  template <LapackMatrix Matrix, MutableComplexLapackVector Vector,
            MutableComplexLapackMatrix Eigenvectors>
    requires LapackEigenvalueVector<Matrix, Vector> && LapackEigenvectorMatrix<Matrix, Eigenvectors>
  void eig_right_vectors(Matrix a, Vector w, Eigenvectors vectors) {
    detail::check_lapack_info("geev", lowlevel::geev_right_vectors(a, w, vectors), a, w, vectors);
  }

  /**
   * @brief Diagonalize a real symmetric tridiagonal matrix with checked host LAPACK diagnostics.
   *
   * The diagonal vector is overwritten with eigenvalues. The off-diagonal vector is LAPACK
   * workspace and is not preserved.
   */
  template <MutableRealLapackVector Diagonal, MutableRealLapackVector OffDiagonal>
    requires mdspan_concepts::SameElementType<Diagonal, OffDiagonal>
  void symmetric_tridiagonal_eigh_values(Diagonal diagonal, OffDiagonal offdiagonal) {
    detail::check_lapack_info("stev", lowlevel::stev_values(diagonal, offdiagonal), diagonal,
                              offdiagonal);
  }

  /**
   * @brief Invert a square matrix in place with checked host LAPACK diagnostics.
   *
   * The wrapper has row-major logical semantics. Internally LAPACK factorizes the transposed
   * column-major view, and the resulting inverse is again represented as a row-major logical
   * matrix.
   */
  template <MutableLapackMatrix Matrix>
  void inverse_inplace(Matrix a) {
    detail::check_lapack_info("getrf/getri", lowlevel::getri_inplace(a), a);
  }

  /**
   * @brief Compute a thin QR factorization of a row-major matrix.
   */
  template <LapackMatrix Matrix, MutableLapackMatrix QMatrix, MutableLapackMatrix RMatrix>
    requires mdspan_concepts::SameElementType<Matrix, QMatrix, RMatrix>
  void qr(Matrix a, QMatrix q, RMatrix r) {
    detail::check_lapack_info("qr", lowlevel::qr(a, q, r), a, q, r);
  }

  /**
   * @brief Compute a thin LQ factorization of a row-major matrix.
   */
  template <LapackMatrix Matrix, MutableLapackMatrix LMatrix, MutableLapackMatrix QMatrix>
    requires mdspan_concepts::SameElementType<Matrix, LMatrix, QMatrix>
  void lq(Matrix a, LMatrix l, QMatrix q) {
    detail::check_lapack_info("lq", lowlevel::lq(a, l, q), a, l, q);
  }

}  // namespace cytnx::lapack

#endif  // BACKEND_TORCH

#endif  // CYTNX_BACKEND_LAPACK_MDSPAN_HPP_
