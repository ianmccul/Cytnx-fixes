#include "gtest/gtest.h"

#include "cytnx.hpp"

using namespace cytnx;

namespace {

  Tensor MakeInvertibleMatrix(const unsigned int dtype) {
    Tensor matrix = zeros({2, 2}, dtype, Device.cpu);
    matrix.at({0, 0}) = 2.0;
    matrix.at({0, 1}) = 0.25;
    matrix.at({1, 0}) = 0.5;
    matrix.at({1, 1}) = 3.0;
    return matrix;
  }

  Tensor MakeHermitianMatrix(const unsigned int dtype) {
    Tensor matrix = zeros({2, 2}, dtype, Device.cpu);
    matrix.at({0, 0}) = 2.0;
    matrix.at({1, 1}) = 3.0;
    if (Type.is_complex(dtype)) {
      matrix.at({0, 1}) = cytnx_complex128(0.25, 0.5);
      matrix.at({1, 0}) = cytnx_complex128(0.25, -0.5);
    } else {
      matrix.at({0, 1}) = 0.25;
      matrix.at({1, 0}) = 0.25;
    }
    return matrix;
  }

  struct DtypeCase {
    unsigned int input;
    unsigned int real_output;
    unsigned int complex_output;
    unsigned int real_part_output;
  };

  class LinalgDtypeSmoke : public testing::TestWithParam<DtypeCase> {};

}  // namespace

TEST_P(LinalgDtypeSmoke, SvdUsesExpectedWorkingDtypes) {
  const DtypeCase param = GetParam();
  std::vector<Tensor> out = linalg::Svd(MakeInvertibleMatrix(param.input));

  ASSERT_EQ(out.size(), 3);
  EXPECT_EQ(out[0].dtype(), param.real_part_output);
  EXPECT_EQ(out[1].dtype(), param.real_output);
  EXPECT_EQ(out[2].dtype(), param.real_output);
}

TEST_P(LinalgDtypeSmoke, EighUsesExpectedWorkingDtypes) {
  const DtypeCase param = GetParam();
  std::vector<Tensor> out = linalg::Eigh(MakeHermitianMatrix(param.input));

  ASSERT_EQ(out.size(), 2);
  EXPECT_EQ(out[0].dtype(), param.real_part_output);
  EXPECT_EQ(out[1].dtype(), param.real_output);
}

TEST_P(LinalgDtypeSmoke, EigUsesComplexWorkingDtypes) {
  const DtypeCase param = GetParam();
  std::vector<Tensor> out = linalg::Eig(MakeInvertibleMatrix(param.input));

  ASSERT_EQ(out.size(), 2);
  EXPECT_EQ(out[0].dtype(), param.complex_output);
  EXPECT_EQ(out[1].dtype(), param.complex_output);
}

TEST_P(LinalgDtypeSmoke, QrUsesExpectedWorkingDtypes) {
  const DtypeCase param = GetParam();
  std::vector<Tensor> out = linalg::Qr(MakeInvertibleMatrix(param.input));

  ASSERT_EQ(out.size(), 2);
  EXPECT_EQ(out[0].dtype(), param.real_output);
  EXPECT_EQ(out[1].dtype(), param.real_output);
}

TEST_P(LinalgDtypeSmoke, LqUsesExpectedWorkingDtypes) {
  const DtypeCase param = GetParam();
  std::vector<Tensor> out = linalg::Lq(MakeInvertibleMatrix(param.input));

  ASSERT_EQ(out.size(), 2);
  EXPECT_EQ(out[0].dtype(), param.real_output);
  EXPECT_EQ(out[1].dtype(), param.real_output);
}

TEST_P(LinalgDtypeSmoke, InvMUsesExpectedWorkingDtype) {
  const DtypeCase param = GetParam();
  Tensor out = linalg::InvM(MakeInvertibleMatrix(param.input));

  EXPECT_EQ(out.dtype(), param.real_output);
}

TEST_P(LinalgDtypeSmoke, ExpMPreservesExpectedPrecision) {
  const DtypeCase param = GetParam();
  Tensor out = linalg::ExpM(MakeInvertibleMatrix(param.input), 1.0);

  EXPECT_EQ(out.dtype(), param.real_output);
}

TEST_P(LinalgDtypeSmoke, ExpMReturnsComplexForComplexScale) {
  const DtypeCase param = GetParam();
  Tensor out = linalg::ExpM(MakeInvertibleMatrix(param.input), cytnx_complex128(0.0, 1.0));

  EXPECT_EQ(out.dtype(), param.complex_output);
}

TEST_P(LinalgDtypeSmoke, ExpHPreservesExpectedPrecisionForRealScale) {
  const DtypeCase param = GetParam();
  Tensor out = linalg::ExpH(MakeHermitianMatrix(param.input), 1.0);

  EXPECT_EQ(out.dtype(), param.real_output);
}

TEST_P(LinalgDtypeSmoke, ExpHReturnsComplexForComplexScale) {
  const DtypeCase param = GetParam();
  Tensor out = linalg::ExpH(MakeHermitianMatrix(param.input), cytnx_complex128(0.0, 1.0));

  EXPECT_EQ(out.dtype(), param.complex_output);
}

INSTANTIATE_TEST_SUITE_P(
  PracticalInputs, LinalgDtypeSmoke,
  testing::Values(DtypeCase{Type.Float, Type.Float, Type.ComplexFloat, Type.Float},
                  DtypeCase{Type.Double, Type.Double, Type.ComplexDouble, Type.Double},
                  DtypeCase{Type.ComplexFloat, Type.ComplexFloat, Type.ComplexFloat, Type.Float},
                  DtypeCase{Type.ComplexDouble, Type.ComplexDouble, Type.ComplexDouble,
                            Type.Double},
                  DtypeCase{Type.Int32, Type.Double, Type.ComplexDouble, Type.Double}));
