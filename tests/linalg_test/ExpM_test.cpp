#include "ExpM_test.h"

TEST(ExpM, ExpM_test) {
  // CompareWithScipy
  std::complex<double> t_i_e[4][4] = {{{-2.0, 0}, {0, 0}, {0, 0}, {-1, 0}},
                                      {{0, 0}, {0, 0}, {-1, 0}, {0, 0}},
                                      {{0, 0}, {-1, 0}, {0, 0}, {0, 0}},
                                      {{-1, 0}, {0, 0}, {0, 0}, {2, 0}}};
  std::complex<double> t_f_e[4][4] = {{{1.10646, 0}, {0, 0}, {0, 0}, {5.01042e-2, 0}},
                                      {{0, 0}, {1.00125, 0}, {5.00208e-2, 0}, {0, 0}},
                                      {{0, 0}, {5.00208e-02, 0}, {1.00125, 0}, {0, 0}},
                                      {{5.01042e-2, 0}, {0, 0}, {0, 0}, {9.06048e-1, 0}}};
  Tensor t_i = zeros(16, Type.ComplexDouble);
  t_i.reshape_(4, 4);
  for (int i = 0; i < 16; i++) {
    int x = i / 4, y = i % 4;
    t_i(x, y) = t_i_e[x][y];
  }
  // std::cout<<t_i<<std::endl;
  // t_i.Mul_(std::complex<double>(0,1));
  double dt = 0.05;
  Tensor t_f = linalg::ExpM(t_i, -dt);
  // std::cout<<t_f<<std::endl;
  for (int i = 0; i < 16; i++) {
    int x = i / 4, y = i % 4;
    // std::cout<<std::fabs(static_cast<double>(t_f(x,y).item().real())-t_f_e[x][y].real())<<std::endl;
    // std::cout<<std::fabs(static_cast<double>(t_f(x,y).item().imag())-t_f_e[x][y].imag())<<std::endl;
    EXPECT_TRUE(std::fabs(static_cast<double>(t_f(x, y).item().real()) - t_f_e[x][y].real()) <
                1e-5);
    EXPECT_TRUE(std::fabs(static_cast<double>(t_f(x, y).item().imag()) - t_f_e[x][y].imag()) <
                1e-5);
  }
}

TEST(ExpM, NilpotentJordanBlock) {
  Tensor t_i = zeros({2, 2}, Type.Double);
  t_i.at<double>({0, 1}) = 1.0;

  Tensor t_f = linalg::ExpM(t_i, 1.0);

  EXPECT_EQ(t_f.dtype(), Type.Double);
  EXPECT_NEAR(t_f.at<double>({0, 0}), 1.0, 1e-12);
  EXPECT_NEAR(t_f.at<double>({0, 1}), 1.0, 1e-12);
  EXPECT_NEAR(t_f.at<double>({1, 0}), 0.0, 1e-12);
  EXPECT_NEAR(t_f.at<double>({1, 1}), 1.0, 1e-12);
}

TEST(ExpM, PreservesFloatForRealInput) {
  Tensor t_i = zeros({2, 2}, Type.Float);
  t_i.at<float>({0, 1}) = 1.0f;

  Tensor t_f = linalg::ExpM(t_i, 1.0);

  EXPECT_EQ(t_f.dtype(), Type.Float);
  EXPECT_NEAR(t_f.at<float>({0, 0}), 1.0f, 1e-6);
  EXPECT_NEAR(t_f.at<float>({0, 1}), 1.0f, 1e-6);
  EXPECT_NEAR(t_f.at<float>({1, 0}), 0.0f, 1e-6);
  EXPECT_NEAR(t_f.at<float>({1, 1}), 1.0f, 1e-6);
}

TEST(ExpM, PreservesComplexFloatForComplexInput) {
  Tensor t_i = zeros({2, 2}, Type.ComplexFloat);
  t_i.at<cytnx_complex64>({0, 1}) = cytnx_complex64(1.0f, 0.0f);

  Tensor t_f = linalg::ExpM(t_i, cytnx_complex64(0.0f, 1.0f));

  EXPECT_EQ(t_f.dtype(), Type.ComplexFloat);
  EXPECT_NEAR(t_f.at<cytnx_complex64>({0, 0}).real(), 1.0f, 1e-6);
  EXPECT_NEAR(t_f.at<cytnx_complex64>({0, 0}).imag(), 0.0f, 1e-6);
  EXPECT_NEAR(t_f.at<cytnx_complex64>({0, 1}).real(), 0.0f, 1e-6);
  EXPECT_NEAR(t_f.at<cytnx_complex64>({0, 1}).imag(), 1.0f, 1e-6);
  EXPECT_NEAR(t_f.at<cytnx_complex64>({1, 0}).real(), 0.0f, 1e-6);
  EXPECT_NEAR(t_f.at<cytnx_complex64>({1, 0}).imag(), 0.0f, 1e-6);
  EXPECT_NEAR(t_f.at<cytnx_complex64>({1, 1}).real(), 1.0f, 1e-6);
  EXPECT_NEAR(t_f.at<cytnx_complex64>({1, 1}).imag(), 0.0f, 1e-6);
}
