#include "gtest/gtest.h"

#include <cmath>

#include "cytnx.hpp"

using namespace cytnx;

namespace {
  constexpr double kFloatTol = 1.0e-6;
  constexpr double kDoubleTol = 1.0e-12;

  Tensor MakeVector(const unsigned int dtype) {
    Tensor tensor({2}, dtype, Device.cpu);
    tensor.at({0}) = 0;
    tensor.at({1}) = 1;
    return tensor;
  }

  Tensor MakeComplexVector(const unsigned int dtype) {
    Tensor tensor({2}, dtype, Device.cpu);
    tensor.at({0}) = 0;
    tensor.at({1}) = cytnx_complex128(0.0, 1.0);
    return tensor;
  }
}  // namespace

TEST(ExpTest, ExpPreservesFloatPrecision) {
  Tensor input = MakeVector(Type.Float);

  Tensor output = linalg::Exp(input);

  EXPECT_EQ(output.dtype(), Type.Float);
  EXPECT_NEAR(output.at<float>({0}), 1.0, kFloatTol);
  EXPECT_NEAR(output.at<float>({1}), std::exp(1.0f), kFloatTol);
}

TEST(ExpTest, ExpPreservesComplexFloatPrecision) {
  Tensor input = MakeComplexVector(Type.ComplexFloat);

  Tensor output = linalg::Exp(input);

  EXPECT_EQ(output.dtype(), Type.ComplexFloat);
  const auto got_zero = output.at<cytnx_complex64>({0});
  const auto got_i = output.at<cytnx_complex64>({1});
  EXPECT_NEAR(got_zero.real(), 1.0, kFloatTol);
  EXPECT_NEAR(got_zero.imag(), 0.0, kFloatTol);
  EXPECT_NEAR(got_i.real(), std::cos(1.0f), kFloatTol);
  EXPECT_NEAR(got_i.imag(), std::sin(1.0f), kFloatTol);
}

TEST(ExpTest, ExpPromotesIntegerInputToDouble) {
  Tensor input = MakeVector(Type.Int32);

  Tensor output = linalg::Exp(input);

  EXPECT_EQ(output.dtype(), Type.Double);
  EXPECT_NEAR(output.at<double>({0}), 1.0, kDoubleTol);
  EXPECT_NEAR(output.at<double>({1}), std::exp(1.0), kDoubleTol);
}
