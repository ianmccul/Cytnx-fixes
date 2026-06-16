#include <gtest/gtest.h>

#include "Generator.hpp"
#include "TensorT_traits.hpp"
#include "backend/lapack_mdspan.hpp"
#include "linalg_mdspan.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#ifndef BACKEND_TORCH

namespace {

  template <class T>
  using matrix_view = cytnx::stdex::mdspan<T, cytnx::stdex::dextents<std::size_t, 2>>;

  template <class T>
  using cuda_matrix_view =
    cytnx::stdex::mdspan<T, cytnx::stdex::dextents<std::size_t, 2>, cytnx::stdex::layout_right,
                         cytnx::stdex::cuda_accessor<T>>;

  template <class T>
  using vector_view = cytnx::stdex::mdspan<T, cytnx::stdex::dextents<std::size_t, 1>>;

  static_assert(cytnx::lapack::RealLapackScalar<float>);
  static_assert(cytnx::lapack::RealLapackScalar<double>);
  static_assert(cytnx::lapack::ComplexLapackScalar<std::complex<float>>);
  static_assert(cytnx::lapack::ComplexLapackScalar<std::complex<double>>);
  static_assert(cytnx::lapack::LapackMatrix<matrix_view<double>>);
  static_assert(cytnx::lapack::LapackMatrix<matrix_view<const double>>);
  static_assert(cytnx::lapack::MutableLapackMatrix<matrix_view<double>>);
  static_assert(!cytnx::lapack::MutableLapackMatrix<matrix_view<const double>>);
  static_assert(!cytnx::lapack::LapackMatrix<cuda_matrix_view<double>>);
  static_assert(cytnx::lapack::RealLapackMatrix<matrix_view<double>>);
  static_assert(cytnx::lapack::RealLapackMatrix<matrix_view<const double>>);
  static_assert(cytnx::lapack::ComplexLapackMatrix<matrix_view<std::complex<double>>>);
  static_assert(cytnx::lapack::LapackVector<vector_view<float>>);
  static_assert(cytnx::lapack::RealLapackVector<vector_view<double>>);
  static_assert(cytnx::lapack::RealLapackVector<vector_view<const double>>);
  static_assert(cytnx::lapack::MutableRealLapackVector<vector_view<double>>);
  static_assert(!cytnx::lapack::MutableRealLapackVector<vector_view<const double>>);
  static_assert(cytnx::lapack::ComplexLapackVector<vector_view<std::complex<float>>>);
  static_assert(cytnx::lapack::MutableComplexLapackMatrix<matrix_view<std::complex<float>>>);
  static_assert(
    cytnx::mdspan_concepts::SameElementType<matrix_view<const double>, matrix_view<double>>);
  static_assert(cytnx::mdspan_concepts::SameElementType<matrix_view<std::complex<double>>,
                                                        matrix_view<std::complex<double>>,
                                                        matrix_view<std::complex<double>>>);
  static_assert(cytnx::mdspan_concepts::SameElementType<
                vector_view<double>,
                cytnx::mdspan_concepts::RealElementOf<matrix_view<std::complex<double>>>>);

  struct DispatchA {};
  struct DispatchB {};
  struct DispatchC {};

  struct DispatchKernel {};
  struct NamedDispatchKernel {
    static constexpr std::string_view name = "dispatch_test";
  };
  struct MdspanDispatchKernel {
    static constexpr std::string_view name = "mdspan_dispatch_test";
  };
  struct DispatchRvalueKernel {};

  void run_kernel(DispatchKernel, DispatchA &, DispatchB &) {}
  void run_kernel(NamedDispatchKernel, DispatchA &, DispatchB &) {}
  void run_kernel(MdspanDispatchKernel, matrix_view<double> &, vector_view<double> &) {}
  void run_kernel(DispatchRvalueKernel, DispatchA &&) {}

  template <class T>
  struct is_std_complex : std::false_type {};

  template <class T>
  struct is_std_complex<std::complex<T>> : std::true_type {};

  template <class T>
  auto conjugate_scalar(const T &value) {
    if constexpr (is_std_complex<T>::value) {
      return std::conj(value);
    } else {
      return value;
    }
  }

  template <class T>
  bool has_eigenvalue_near(const std::vector<std::complex<T>> &values, std::complex<T> expected,
                           T tolerance) {
    for (const auto &value : values) {
      if (std::abs(value - expected) <= tolerance) return true;
    }
    return false;
  }

  template <class T, class Real>
  void expect_qr_reconstructs(const std::vector<T> &original, const std::vector<T> &q,
                              const std::vector<T> &r, std::size_t rows, std::size_t cols,
                              Real tolerance) {
    const std::size_t k = std::min(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        T reconstructed{};
        for (std::size_t p = 0; p < k; ++p) {
          reconstructed += q[i * k + p] * r[p * cols + j];
        }
        EXPECT_NEAR(std::abs(reconstructed - original[i * cols + j]), Real{}, tolerance);
      }
    }
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        T inner{};
        for (std::size_t p = 0; p < rows; ++p) {
          inner += conjugate_scalar(q[p * k + i]) * q[p * k + j];
        }
        EXPECT_NEAR(std::abs(inner - (i == j ? T{1} : T{})), Real{}, tolerance);
      }
    }
  }

  template <class T, class Real>
  void expect_lq_reconstructs(const std::vector<T> &original, const std::vector<T> &l,
                              const std::vector<T> &q, std::size_t rows, std::size_t cols,
                              Real tolerance) {
    const std::size_t k = std::min(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        T reconstructed{};
        for (std::size_t p = 0; p < k; ++p) {
          reconstructed += l[i * k + p] * q[p * cols + j];
        }
        EXPECT_NEAR(std::abs(reconstructed - original[i * cols + j]), Real{}, tolerance);
      }
    }
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        T inner{};
        for (std::size_t p = 0; p < cols; ++p) {
          inner += q[i * cols + p] * conjugate_scalar(q[j * cols + p]);
        }
        EXPECT_NEAR(std::abs(inner - (i == j ? T{1} : T{})), Real{}, tolerance);
      }
    }
  }

  template <class MatrixScalar, class VectorScalar, class EigenvalueScalar, class Real>
  void expect_right_eigenvectors(const std::vector<MatrixScalar> &matrix,
                                 const std::vector<EigenvalueScalar> &values,
                                 const std::vector<VectorScalar> &vectors, std::size_t n,
                                 Real tolerance) {
    for (std::size_t eig = 0; eig < n; ++eig) {
      for (std::size_t row = 0; row < n; ++row) {
        std::complex<Real> applied{};
        for (std::size_t col = 0; col < n; ++col) {
          applied += matrix[row * n + col] * vectors[eig * n + col];
        }
        const auto expected = values[eig] * vectors[eig * n + row];
        EXPECT_NEAR(std::abs(applied - expected), Real{}, tolerance);
      }
    }
  }

  static_assert(cytnx::Variant<std::variant<DispatchA, DispatchC>>);
  static_assert(
    cytnx::AnyDispatchInvocable<DispatchKernel, std::variant<DispatchC, DispatchA> &, DispatchB &>);
  static_assert(
    !cytnx::AnyDispatchInvocable<DispatchKernel, std::variant<DispatchC> &, DispatchB &>);
  static_assert(cytnx::AnyDispatchInvocable<DispatchKernel, DispatchA &, DispatchB &>);
  static_assert(!cytnx::AnyDispatchInvocable<DispatchKernel, DispatchA, DispatchB>);
  static_assert(
    cytnx::AnyDispatchInvocable<DispatchRvalueKernel, std::variant<DispatchC, DispatchA>>);
  static_assert(
    !cytnx::AnyDispatchInvocable<DispatchRvalueKernel, std::variant<DispatchC, DispatchA> &>);

  TEST(LapackMdspanTest, InvokeKernelReportsActiveAlternativeFailure) {
    std::variant<DispatchC, DispatchA> active_bad = DispatchC{};
    DispatchB second;

    try {
      cytnx::invoke_kernel(NamedDispatchKernel{}, active_bad, second);
      FAIL() << "Expected invoke_kernel to reject the active variant alternative.";
    } catch (const std::logic_error &err) {
      const std::string message = err.what();
      EXPECT_NE(message.find("No matching backend kernel found for dispatch_test"),
                std::string::npos);
      EXPECT_NE(message.find("arguments:"), std::string::npos);
      EXPECT_NE(message.find("arg0:"), std::string::npos);
      EXPECT_NE(message.find("arg1:"), std::string::npos);
    }
  }

  TEST(LapackMdspanTest, InvokeKernelReportsMdspanArgumentMetadata) {
    std::vector<float> bad_matrix_data(4);
    std::vector<double> values_data(2);
    std::variant<matrix_view<float>, matrix_view<double>> active_bad =
      matrix_view<float>(bad_matrix_data.data(), 2, 2);
    vector_view<double> values(values_data.data(), 2);

    try {
      cytnx::invoke_kernel(MdspanDispatchKernel{}, active_bad, values);
      FAIL() << "Expected invoke_kernel to reject the active mdspan alternative.";
    } catch (const std::logic_error &err) {
      const std::string message = err.what();
      EXPECT_NE(message.find("No matching backend kernel found for mdspan_dispatch_test"),
                std::string::npos);
      EXPECT_NE(message.find("rank: 2"), std::string::npos);
      EXPECT_NE(message.find("extents: [2, 2]"), std::string::npos);
      EXPECT_NE(message.find("strides:"), std::string::npos);
      EXPECT_NE(message.find("element_type:"), std::string::npos);
      EXPECT_NE(message.find("layout: layout_right"), std::string::npos);
      EXPECT_NE(message.find("access: host"), std::string::npos);
    }
  }

  TEST(LapackMdspanTest, RowMajorSyevComputesEigenvalues) {
    std::vector<double> a = {
      3.0,
      0.0,
      0.0,
      1.0,
    };
    std::vector<double> w(2);

    const int info = cytnx::lapack::lowlevel::eigh('N', 'U', matrix_view<double>(a.data(), 2, 2),
                                                   vector_view<double>(w.data(), 2));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(w[0], 1.0, 1e-12);
    EXPECT_NEAR(w[1], 3.0, 1e-12);
  }

  TEST(LapackMdspanTest, RowMajorSyevSupportsFloat) {
    std::vector<float> a = {
      5.0F,
      0.0F,
      0.0F,
      2.0F,
    };
    std::vector<float> w(2);

    const int info = cytnx::lapack::lowlevel::eigh('N', 'U', matrix_view<float>(a.data(), 2, 2),
                                                   vector_view<float>(w.data(), 2));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(w[0], 2.0F, 1e-5F);
    EXPECT_NEAR(w[1], 5.0F, 1e-5F);
  }

  TEST(LapackMdspanTest, RowMajorHeevUsesLogicalUpperTriangle) {
    using complex = std::complex<double>;
    std::vector<complex> a = {
      complex{2.0, 0.0},
      complex{0.0, 1.0},
      complex{99.0, 0.0},
      complex{2.0, 0.0},
    };
    std::vector<double> w(2);

    const int info = cytnx::lapack::lowlevel::eigh('N', 'U', matrix_view<complex>(a.data(), 2, 2),
                                                   vector_view<double>(w.data(), 2));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(w[0], 1.0, 1e-12);
    EXPECT_NEAR(w[1], 3.0, 1e-12);
  }

  TEST(LapackMdspanTest, RowMajorGeevComputesRealMatrixEigenvalues) {
    using complex = std::complex<double>;
    std::vector<double> a = {
      0.0,
      -1.0,
      1.0,
      0.0,
    };
    std::vector<complex> w(2);

    const int info = cytnx::lapack::lowlevel::geev_values(matrix_view<double>(a.data(), 2, 2),
                                                          vector_view<complex>(w.data(), 2));

    ASSERT_EQ(info, 0);
    EXPECT_TRUE(has_eigenvalue_near(w, complex{0.0, 1.0}, 1e-12));
    EXPECT_TRUE(has_eigenvalue_near(w, complex{0.0, -1.0}, 1e-12));
  }

  TEST(LapackMdspanTest, RowMajorGeevSupportsComplexMatrices) {
    using complex = std::complex<double>;
    std::vector<complex> a = {
      complex{2.0, 1.0},
      complex{0.0, 0.0},
      complex{0.0, 0.0},
      complex{-1.0, 0.5},
    };
    std::vector<complex> w(2);

    const int info = cytnx::lapack::lowlevel::geev_values(matrix_view<complex>(a.data(), 2, 2),
                                                          vector_view<complex>(w.data(), 2));

    ASSERT_EQ(info, 0);
    EXPECT_TRUE(has_eigenvalue_near(w, complex{2.0, 1.0}, 1e-12));
    EXPECT_TRUE(has_eigenvalue_near(w, complex{-1.0, 0.5}, 1e-12));
  }

  TEST(LapackMdspanTest, StevComputesSymmetricTridiagonalEigenvalues) {
    std::vector<double> diagonal = {2.0, 3.0};
    std::vector<double> offdiagonal = {1.0};

    const int info = cytnx::lapack::lowlevel::stev_values(
      vector_view<double>(diagonal.data(), 2), vector_view<double>(offdiagonal.data(), 1));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(diagonal[0], (5.0 - std::sqrt(5.0)) / 2.0, 1e-12);
    EXPECT_NEAR(diagonal[1], (5.0 + std::sqrt(5.0)) / 2.0, 1e-12);
  }

  TEST(LapackMdspanTest, GetriInvertsRowMajorMatrixInPlace) {
    std::vector<double> a = {
      4.0,
      7.0,
      2.0,
      6.0,
    };

    const int info = cytnx::lapack::lowlevel::getri_inplace(matrix_view<double>(a.data(), 2, 2));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(a[0], 0.6, 1e-12);
    EXPECT_NEAR(a[1], -0.7, 1e-12);
    EXPECT_NEAR(a[2], -0.2, 1e-12);
    EXPECT_NEAR(a[3], 0.4, 1e-12);
  }

  TEST(LapackMdspanTest, GetriSupportsComplexMatrices) {
    using complex = std::complex<double>;
    std::vector<complex> a = {
      complex{1.0, 1.0},
      complex{0.0, 0.0},
      complex{0.0, 0.0},
      complex{2.0, -1.0},
    };

    const int info = cytnx::lapack::lowlevel::getri_inplace(matrix_view<complex>(a.data(), 2, 2));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(std::abs(a[0] - complex{0.5, -0.5}), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(a[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(a[2]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(a[3] - complex{0.4, 0.2}), 0.0, 1e-12);
  }

  TEST(LapackMdspanTest, GetriAcceptsEmptyMatrix) {
    std::vector<double> a;

    const int info = cytnx::lapack::lowlevel::getri_inplace(matrix_view<double>(a.data(), 0, 0));

    EXPECT_EQ(info, 0);
  }

  TEST(LapackMdspanTest, GetriReportsSingularMatrix) {
    std::vector<double> a = {
      1.0,
      2.0,
      2.0,
      4.0,
    };

    const int info = cytnx::lapack::lowlevel::getri_inplace(matrix_view<double>(a.data(), 2, 2));

    EXPECT_GT(info, 0);
  }

  TEST(LapackMdspanTest, RowMajorGesvdComputesSingularValues) {
    std::vector<double> a = {
      3.0, 0.0, 0.0, 0.0, 4.0, 0.0,
    };
    std::vector<double> s(2);

    const int info = cytnx::lapack::lowlevel::gesvd(matrix_view<double>(a.data(), 2, 3),
                                                    vector_view<double>(s.data(), 2));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(s[0], 4.0, 1e-12);
    EXPECT_NEAR(s[1], 3.0, 1e-12);
  }

  TEST(LapackMdspanTest, RowMajorGesvdSupportsComplexFloat) {
    using complex = std::complex<float>;
    std::vector<complex> a = {
      complex{0.0F, 3.0F},
      complex{0.0F, 0.0F},
      complex{0.0F, 0.0F},
      complex{4.0F, 0.0F},
    };
    std::vector<float> s(2);

    const int info = cytnx::lapack::lowlevel::gesvd(matrix_view<complex>(a.data(), 2, 2),
                                                    vector_view<float>(s.data(), 2));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(s[0], 4.0F, 1e-5F);
    EXPECT_NEAR(s[1], 3.0F, 1e-5F);
  }

  TEST(LapackMdspanTest, RowMajorGesddComputesSingularValues) {
    std::vector<double> a = {
      3.0, 0.0, 0.0, 0.0, 4.0, 0.0,
    };
    std::vector<double> s(2);

    const int info = cytnx::lapack::lowlevel::gesdd(matrix_view<double>(a.data(), 2, 3),
                                                    vector_view<double>(s.data(), 2));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(s[0], 4.0, 1e-12);
    EXPECT_NEAR(s[1], 3.0, 1e-12);
  }

  TEST(LapackMdspanTest, RowMajorGesddSupportsComplexFloat) {
    using complex = std::complex<float>;
    std::vector<complex> a = {
      complex{0.0F, 3.0F},
      complex{0.0F, 0.0F},
      complex{0.0F, 0.0F},
      complex{4.0F, 0.0F},
    };
    std::vector<float> s(2);

    const int info = cytnx::lapack::lowlevel::gesdd(matrix_view<complex>(a.data(), 2, 2),
                                                    vector_view<float>(s.data(), 2));

    ASSERT_EQ(info, 0);
    EXPECT_NEAR(s[0], 4.0F, 1e-5F);
    EXPECT_NEAR(s[1], 3.0F, 1e-5F);
  }

  TEST(LapackMdspanTest, RowMajorGesvdComputesThinFactors) {
    const std::vector<double> original = {
      1.0, 2.0, 0.0, 0.0, 1.0, 3.0,
    };
    std::vector<double> a = original;
    std::vector<double> s(2);
    std::vector<double> u(2 * 2);
    std::vector<double> vt(2 * 3);

    cytnx::lapack::svd(matrix_view<double>(a.data(), 2, 3), vector_view<double>(s.data(), 2),
                       matrix_view<double>(u.data(), 2, 2), matrix_view<double>(vt.data(), 2, 3));

    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        double reconstructed = 0.0;
        for (std::size_t k = 0; k < 2; ++k) {
          reconstructed += u[i * 2 + k] * s[k] * vt[k * 3 + j];
        }
        EXPECT_NEAR(reconstructed, original[i * 3 + j], 1e-12);
      }
    }
  }

  TEST(LapackMdspanTest, RowMajorGesddComputesThinFactors) {
    const std::vector<double> original = {
      1.0, 2.0, 0.0, 0.0, 1.0, 3.0,
    };
    std::vector<double> a = original;
    std::vector<double> s(2);
    std::vector<double> u(2 * 2);
    std::vector<double> vt(2 * 3);

    cytnx::lapack::svd_divide_conquer(
      matrix_view<double>(a.data(), 2, 3), vector_view<double>(s.data(), 2),
      matrix_view<double>(u.data(), 2, 2), matrix_view<double>(vt.data(), 2, 3));

    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        double reconstructed = 0.0;
        for (std::size_t k = 0; k < 2; ++k) {
          reconstructed += u[i * 2 + k] * s[k] * vt[k * 3 + j];
        }
        EXPECT_NEAR(reconstructed, original[i * 3 + j], 1e-12);
      }
    }
  }

  TEST(LapackMdspanTest, CheckedSvdValuesWrapperComputesSingularValues) {
    std::vector<double> a = {
      3.0, 0.0, 0.0, 0.0, 4.0, 0.0,
    };
    std::vector<double> s(2);

    cytnx::lapack::svd_values(matrix_view<double>(a.data(), 2, 3),
                              vector_view<double>(s.data(), 2));

    EXPECT_NEAR(s[0], 4.0, 1e-12);
    EXPECT_NEAR(s[1], 3.0, 1e-12);
  }

  TEST(LapackMdspanTest, CheckedGesddValuesWrapperComputesSingularValues) {
    std::vector<double> a = {
      3.0, 0.0, 0.0, 0.0, 4.0, 0.0,
    };
    std::vector<double> s(2);

    cytnx::lapack::svd_divide_conquer_values(matrix_view<double>(a.data(), 2, 3),
                                             vector_view<double>(s.data(), 2));

    EXPECT_NEAR(s[0], 4.0, 1e-12);
    EXPECT_NEAR(s[1], 3.0, 1e-12);
  }

  TEST(LapackMdspanTest, GesddRejectsSmallSingularValueOutput) {
    std::vector<double> a = {
      1.0,
      0.0,
      0.0,
      1.0,
    };
    std::vector<double> s(1);

    try {
      cytnx::lapack::svd_divide_conquer_values(matrix_view<double>(a.data(), 2, 2),
                                               vector_view<double>(s.data(), 1));
      FAIL() << "Expected too-small singular-value output to fail.";
    } catch (const std::logic_error &err) {
      const std::string message = err.what();
      EXPECT_NE(message.find("LAPACK gesdd singular-value output is too small"), std::string::npos);
    }
  }

  TEST(LapackMdspanTest, QrFactorizesTallRealMatrix) {
    const std::vector<double> original = {
      1.0, 2.0, 3.0, 4.0, 5.0, 7.0,
    };
    std::vector<double> a = original;
    std::vector<double> q(3 * 2);
    std::vector<double> r(2 * 2);

    cytnx::lapack::qr(matrix_view<const double>(a.data(), 3, 2),
                      matrix_view<double>(q.data(), 3, 2), matrix_view<double>(r.data(), 2, 2));

    EXPECT_EQ(a, original);
    expect_qr_reconstructs(original, q, r, 3, 2, 1e-12);
  }

  TEST(LapackMdspanTest, QrFactorizesWideFloatMatrix) {
    const std::vector<float> original = {
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 7.0F,
    };
    std::vector<float> a = original;
    std::vector<float> q(2 * 2);
    std::vector<float> r(2 * 3);

    cytnx::qr(matrix_view<const float>(a.data(), 2, 3), matrix_view<float>(q.data(), 2, 2),
              matrix_view<float>(r.data(), 2, 3));

    EXPECT_EQ(a, original);
    expect_qr_reconstructs(original, q, r, 2, 3, 1e-5F);
  }

  TEST(LapackMdspanTest, LqFactorizesTallRealMatrix) {
    const std::vector<double> original = {
      1.0, 2.0, 3.0, 4.0, 5.0, 7.0,
    };
    std::vector<double> a = original;
    std::vector<double> l(3 * 2);
    std::vector<double> q(2 * 2);

    cytnx::lapack::lq(matrix_view<const double>(a.data(), 3, 2),
                      matrix_view<double>(l.data(), 3, 2), matrix_view<double>(q.data(), 2, 2));

    EXPECT_EQ(a, original);
    expect_lq_reconstructs(original, l, q, 3, 2, 1e-12);
  }

  TEST(LapackMdspanTest, LqFactorizesWideFloatMatrix) {
    const std::vector<float> original = {
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 7.0F,
    };
    std::vector<float> a = original;
    std::vector<float> l(2 * 2);
    std::vector<float> q(2 * 3);

    cytnx::lq(matrix_view<const float>(a.data(), 2, 3), matrix_view<float>(l.data(), 2, 2),
              matrix_view<float>(q.data(), 2, 3));

    EXPECT_EQ(a, original);
    expect_lq_reconstructs(original, l, q, 2, 3, 1e-5F);
  }

  TEST(LapackMdspanTest, QrFactorizesComplexMatrix) {
    using complex = std::complex<double>;
    const std::vector<complex> original = {
      complex{1.0, 1.0},  complex{2.0, -1.0}, complex{3.0, 0.5},
      complex{-1.0, 0.0}, complex{4.0, 2.0},  complex{0.0, -3.0},
    };
    std::vector<complex> a = original;
    std::vector<complex> q(3 * 2);
    std::vector<complex> r(2 * 2);

    cytnx::qr(matrix_view<const complex>(a.data(), 3, 2), matrix_view<complex>(q.data(), 3, 2),
              matrix_view<complex>(r.data(), 2, 2));

    EXPECT_EQ(a, original);
    expect_qr_reconstructs(original, q, r, 3, 2, 1e-12);
  }

  TEST(LapackMdspanTest, LqFactorizesComplexFloatMatrix) {
    using complex = std::complex<float>;
    const std::vector<complex> original = {
      complex{1.0F, 0.5F},  complex{-2.0F, 1.0F}, complex{3.0F, 0.0F},
      complex{4.0F, -1.0F}, complex{0.5F, 2.0F},  complex{-1.0F, -0.5F},
    };
    std::vector<complex> a = original;
    std::vector<complex> l(2 * 2);
    std::vector<complex> q(2 * 3);

    cytnx::lq(matrix_view<const complex>(a.data(), 2, 3), matrix_view<complex>(l.data(), 2, 2),
              matrix_view<complex>(q.data(), 2, 3));

    EXPECT_EQ(a, original);
    expect_lq_reconstructs(original, l, q, 2, 3, 1e-5F);
  }

  TEST(LapackMdspanTest, QrRejectsIncompatibleOutputShape) {
    std::vector<double> a = {
      1.0,
      2.0,
      3.0,
      4.0,
    };
    std::vector<double> q(2);
    std::vector<double> r(4);

    try {
      cytnx::qr(matrix_view<double>(a.data(), 2, 2), matrix_view<double>(q.data(), 2, 1),
                matrix_view<double>(r.data(), 2, 2));
      FAIL() << "Expected incompatible QR output shape to fail.";
    } catch (const std::logic_error &err) {
      const std::string message = err.what();
      EXPECT_NE(message.find("LAPACK qr Q output has incompatible shape"), std::string::npos);
    }
  }

  TEST(LapackMdspanTest, CheckedSelfAdjointEighWrapperComputesEigenvalues) {
    std::vector<double> a = {
      3.0,
      0.0,
      0.0,
      1.0,
    };
    std::vector<double> w(2);

    cytnx::lapack::self_adjoint_eigh('N', 'U', matrix_view<double>(a.data(), 2, 2),
                                     vector_view<double>(w.data(), 2));

    EXPECT_NEAR(w[0], 1.0, 1e-12);
    EXPECT_NEAR(w[1], 3.0, 1e-12);
  }

  TEST(LapackMdspanTest, SelfAdjointEighVectorsWritesEigenvectorsAsRows) {
    using complex = std::complex<double>;
    const std::vector<complex> original = {
      complex{2.0, 0.0},
      complex{0.0, 1.0},
      complex{0.0, -1.0},
      complex{2.0, 0.0},
    };
    std::vector<complex> a = original;
    std::vector<double> w(2);
    std::vector<complex> vectors(2 * 2);

    cytnx::self_adjoint_eigh_vectors('U', matrix_view<complex>(a.data(), 2, 2),
                                     vector_view<double>(w.data(), 2),
                                     matrix_view<complex>(vectors.data(), 2, 2));

    EXPECT_NEAR(w[0], 1.0, 1e-12);
    EXPECT_NEAR(w[1], 3.0, 1e-12);
    expect_right_eigenvectors(original, w, vectors, 2, 1e-12);
  }

  TEST(LapackMdspanTest, CheckedGeevWrapperComputesEigenvalues) {
    using complex = std::complex<double>;
    std::vector<double> a = {
      0.0,
      -2.0,
      2.0,
      0.0,
    };
    std::vector<complex> w(2);

    cytnx::lapack::eig_values(matrix_view<double>(a.data(), 2, 2),
                              vector_view<complex>(w.data(), 2));

    EXPECT_TRUE(has_eigenvalue_near(w, complex{0.0, 2.0}, 1e-12));
    EXPECT_TRUE(has_eigenvalue_near(w, complex{0.0, -2.0}, 1e-12));
  }

  TEST(LapackMdspanTest, EigRightVectorsPromotesRealInputToComplex) {
    using complex = std::complex<double>;
    const std::vector<double> original = {
      0.0,
      -2.0,
      2.0,
      0.0,
    };
    std::vector<double> a = original;
    std::vector<complex> w(2);
    std::vector<complex> vectors(2 * 2);

    cytnx::eig_right_vectors(matrix_view<const double>(a.data(), 2, 2),
                             vector_view<complex>(w.data(), 2),
                             matrix_view<complex>(vectors.data(), 2, 2));

    EXPECT_EQ(a, original);
    EXPECT_TRUE(has_eigenvalue_near(w, complex{0.0, 2.0}, 1e-12));
    EXPECT_TRUE(has_eigenvalue_near(w, complex{0.0, -2.0}, 1e-12));
    expect_right_eigenvectors(original, w, vectors, 2, 1e-12);
  }

  TEST(LapackMdspanTest, EigRightVectorsSupportsComplexInput) {
    using complex = std::complex<double>;
    const std::vector<complex> original = {
      complex{1.0, 2.0},
      complex{0.0, 0.0},
      complex{0.0, 0.0},
      complex{3.0, -1.0},
    };
    std::vector<complex> w(2);
    std::vector<complex> vectors(2 * 2);

    cytnx::eig_right_vectors(matrix_view<const complex>(original.data(), 2, 2),
                             vector_view<complex>(w.data(), 2),
                             matrix_view<complex>(vectors.data(), 2, 2));

    EXPECT_TRUE(has_eigenvalue_near(w, complex{1.0, 2.0}, 1e-12));
    EXPECT_TRUE(has_eigenvalue_near(w, complex{3.0, -1.0}, 1e-12));
    expect_right_eigenvectors(original, w, vectors, 2, 1e-12);
  }

  TEST(LapackMdspanTest, CheckedStevWrapperComputesSymmetricTridiagonalEigenvalues) {
    std::vector<float> diagonal = {2.0F, 3.0F};
    std::vector<float> offdiagonal = {1.0F};

    cytnx::lapack::symmetric_tridiagonal_eigh_values(vector_view<float>(diagonal.data(), 2),
                                                     vector_view<float>(offdiagonal.data(), 1));

    EXPECT_NEAR(diagonal[0], (5.0F - std::sqrt(5.0F)) / 2.0F, 1e-5F);
    EXPECT_NEAR(diagonal[1], (5.0F + std::sqrt(5.0F)) / 2.0F, 1e-5F);
  }

  TEST(LapackMdspanTest, CheckedInverseWrapperInvertsMatrix) {
    std::vector<float> a = {
      4.0F,
      7.0F,
      2.0F,
      6.0F,
    };

    cytnx::lapack::inverse_inplace(matrix_view<float>(a.data(), 2, 2));

    EXPECT_NEAR(a[0], 0.6F, 1e-5F);
    EXPECT_NEAR(a[1], -0.7F, 1e-5F);
    EXPECT_NEAR(a[2], -0.2F, 1e-5F);
    EXPECT_NEAR(a[3], 0.4F, 1e-5F);
  }

  TEST(LapackMdspanTest, CheckedInverseWrapperRejectsSingularMatrix) {
    std::vector<double> a = {
      1.0,
      2.0,
      2.0,
      4.0,
    };

    try {
      cytnx::lapack::inverse_inplace(matrix_view<double>(a.data(), 2, 2));
      FAIL() << "Expected singular matrix inversion to fail.";
    } catch (const std::logic_error &err) {
      const std::string message = err.what();
      EXPECT_NE(message.find("LAPACK getrf/getri failed"), std::string::npos);
      EXPECT_NE(message.find("info = "), std::string::npos);
    }
  }

  TEST(LapackMdspanTest, InverseWrapperRejectsNonSquareMatrix) {
    std::vector<double> a(6);

    try {
      cytnx::lapack::inverse_inplace(matrix_view<double>(a.data(), 2, 3));
      FAIL() << "Expected non-square matrix inversion to fail.";
    } catch (const std::logic_error &err) {
      const std::string message = err.what();
      EXPECT_NE(message.find("LAPACK getri input must be square"), std::string::npos);
    }
  }

  TEST(LapackMdspanTest, VariantSvdValuesDispatchesTensorTAlternatives) {
    cytnx::Tensor a = cytnx::zeros({2, 3}, cytnx::Type.Double);
    a.at<double>({0, 0}) = 3.0;
    a.at<double>({1, 1}) = 4.0;
    cytnx::Tensor s = cytnx::zeros({2}, cytnx::Type.Double);

    cytnx::RealTensor<2> a_view = cytnx::make_right_tensor_t<double, 2>(a);
    cytnx::RealTensor<1> s_view = cytnx::make_right_tensor_t<double, 1>(s);

    cytnx::svd_values(a_view, s_view);

    EXPECT_NEAR(s.at<double>({0}), 4.0, 1e-12);
    EXPECT_NEAR(s.at<double>({1}), 3.0, 1e-12);
  }

  TEST(LapackMdspanTest, SvdValuesDispatchesMixedFixedAndVariantArguments) {
    cytnx::Tensor a = cytnx::zeros({2, 3}, cytnx::Type.Double);
    a.at<double>({0, 0}) = 3.0;
    a.at<double>({1, 1}) = 4.0;
    cytnx::Tensor s = cytnx::zeros({2}, cytnx::Type.Double);

    auto a_view = cytnx::make_right_tensor_t<double, 2>(a);
    cytnx::RealTensor<1> s_view = cytnx::make_right_tensor_t<double, 1>(s);

    cytnx::svd_values(a_view, s_view);

    EXPECT_NEAR(s.at<double>({0}), 4.0, 1e-12);
    EXPECT_NEAR(s.at<double>({1}), 3.0, 1e-12);
  }

  TEST(LapackMdspanTest, InverseInplaceDispatchesTensorTVariantAndMutatesTensor) {
    cytnx::Tensor a = cytnx::zeros({2, 2}, cytnx::Type.Double);
    a.at<double>({0, 0}) = 4.0;
    a.at<double>({0, 1}) = 7.0;
    a.at<double>({1, 0}) = 2.0;
    a.at<double>({1, 1}) = 6.0;

    cytnx::NumericTensor<2> a_view = cytnx::make_right_tensor_t<double, 2>(a);

    cytnx::inverse_inplace(a_view);

    EXPECT_NEAR(a.at<double>({0, 0}), 0.6, 1e-12);
    EXPECT_NEAR(a.at<double>({0, 1}), -0.7, 1e-12);
    EXPECT_NEAR(a.at<double>({1, 0}), -0.2, 1e-12);
    EXPECT_NEAR(a.at<double>({1, 1}), 0.4, 1e-12);
  }

  TEST(LapackMdspanTest, PublicSvdValuesAcceptsRawHostMdspans) {
    std::vector<double> a = {
      1.0,
      2.0,
      3.0,
      4.0,
    };
    std::vector<double> s(2);

    cytnx::svd_values(matrix_view<double>(a.data(), 2, 2), vector_view<double>(s.data(), 2));

    EXPECT_NEAR(s[0], 5.464985704219043, 1e-12);
    EXPECT_NEAR(s[1], 0.365966190626257, 1e-12);
  }

  TEST(LapackMdspanTest, PublicGesddValuesAcceptsRawHostMdspans) {
    std::vector<double> a = {
      1.0,
      2.0,
      3.0,
      4.0,
    };
    std::vector<double> s(2);

    cytnx::svd_divide_conquer_values(matrix_view<double>(a.data(), 2, 2),
                                     vector_view<double>(s.data(), 2));

    EXPECT_NEAR(s[0], 5.464985704219043, 1e-12);
    EXPECT_NEAR(s[1], 0.365966190626257, 1e-12);
  }

  TEST(LapackMdspanTest, PublicEigValuesAcceptsRawHostMdspans) {
    using complex = std::complex<double>;
    std::vector<double> a = {
      0.0,
      -3.0,
      3.0,
      0.0,
    };
    std::vector<complex> w(2);

    cytnx::eig_values(matrix_view<double>(a.data(), 2, 2), vector_view<complex>(w.data(), 2));

    EXPECT_TRUE(has_eigenvalue_near(w, complex{0.0, 3.0}, 1e-12));
    EXPECT_TRUE(has_eigenvalue_near(w, complex{0.0, -3.0}, 1e-12));
  }

  TEST(LapackMdspanTest, PublicStevValuesAcceptsRawHostMdspans) {
    std::vector<double> diagonal = {2.0, 3.0};
    std::vector<double> offdiagonal = {1.0};

    cytnx::symmetric_tridiagonal_eigh_values(vector_view<double>(diagonal.data(), 2),
                                             vector_view<double>(offdiagonal.data(), 1));

    EXPECT_NEAR(diagonal[0], (5.0 - std::sqrt(5.0)) / 2.0, 1e-12);
    EXPECT_NEAR(diagonal[1], (5.0 + std::sqrt(5.0)) / 2.0, 1e-12);
  }

  TEST(LapackMdspanTest, PublicInverseInplaceAcceptsRawHostMdspan) {
    std::vector<double> a = {
      4.0,
      7.0,
      2.0,
      6.0,
    };

    cytnx::inverse_inplace(matrix_view<double>(a.data(), 2, 2));

    EXPECT_NEAR(a[0], 0.6, 1e-12);
    EXPECT_NEAR(a[1], -0.7, 1e-12);
    EXPECT_NEAR(a[2], -0.2, 1e-12);
    EXPECT_NEAR(a[3], 0.4, 1e-12);
  }

}  // namespace

#endif  // BACKEND_TORCH
