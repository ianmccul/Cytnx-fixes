#include <gtest/gtest.h>

#include "Generator.hpp"
#include "TensorT_traits.hpp"
#include "backend/lapack_mdspan.hpp"
#include "linalg_mdspan.hpp"

#include <cmath>
#include <complex>
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
  static_assert(!cytnx::lapack::LapackMatrix<cuda_matrix_view<double>>);
  static_assert(cytnx::lapack::RealLapackMatrix<matrix_view<double>>);
  static_assert(cytnx::lapack::ComplexLapackMatrix<matrix_view<std::complex<double>>>);
  static_assert(cytnx::lapack::LapackVector<vector_view<float>>);
  static_assert(cytnx::lapack::RealLapackVector<vector_view<double>>);
  static_assert(cytnx::lapack::ComplexLapackVector<vector_view<std::complex<float>>>);
  static_assert(cytnx::mdspan_concepts::SameElementType<matrix_view<std::complex<double>>,
                                                        matrix_view<std::complex<double>>,
                                                        matrix_view<std::complex<double>>>);
  static_assert(cytnx::mdspan_concepts::SameElementType<
                vector_view<double>,
                cytnx::mdspan_concepts::RealElementOf<matrix_view<std::complex<double>>>>);

  struct DispatchA {};
  struct DispatchB {};
  struct DispatchC {};

  struct DispatchCallable {
    void operator()(DispatchA &, DispatchB &) const {}
  };

  static_assert(cytnx::Variant<std::variant<DispatchA, DispatchC>>);
  static_assert(
    cytnx::AnyDispatchInvocable<DispatchCallable, std::variant<DispatchC, DispatchA>, DispatchB>);
  static_assert(!cytnx::AnyDispatchInvocable<DispatchCallable, std::variant<DispatchC>, DispatchB>);

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

}  // namespace

#endif  // BACKEND_TORCH
