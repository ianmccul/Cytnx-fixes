#include "gtest/gtest.h"

#include <cmath>
#include <complex>
#include <limits>
#include <string>

#include "test_tools.h"
#include "cytnx.hpp"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace Lanczos_Exp_Ut_Test {

  UniTensor CreateOneSiteEffHam(const int d, const int D, const unsigned int dypte = Type.Double,
                                const int device = Device.cpu);
  UniTensor CreateA(const int d, const int D, const unsigned int dtype = Type.Double,
                    const int device = Device.cpu);
  UniTensor GetAns(const UniTensor& HEff, const UniTensor& Tin, const Scalar& tau);
  Scalar Dot(const UniTensor& A, const UniTensor& B) { return Contract(A.Dagger(), B).item(); }
  class OneSiteOp : public LinOp {
   public:
    OneSiteOp(const int d, const int D, const unsigned int dtype = Type.Double,
              const int& device = Device.cpu)
        : LinOp("mv", D * d * D, dtype, device) {
      EffH = CreateOneSiteEffHam(d, D, dtype, device);
    }
    UniTensor EffH;

    /*
     *         |-|--"vil" "pi" "vir"--|-|                        |-|--"vil" "pi" "vir"--|-|
     *         | |         +          | |             "po"       | |         +          | |
     *         |L|- -------O----------|R|  dot         |       = |L|- -------O----------|R|
     *         | |         +          | |       "vol"--A--"vor"  | |         +          | |
     *         |_|--"vol" "po" "vor"--|_|                        |_|---------A----------|_|
     *
     * Then relabel ["vil", "pi", "vir"] -> ["vol", "po", "vor"]
     *
     * "vil":virtual in bond left
     * "po":physical out bond
     */
    UniTensor matvec_impl(const UniTensor& A) override {
      auto tmp = Contract(EffH, A);
      tmp.permute_({"vil", "pi", "vir"}, 1);
      tmp.relabel_(A.labels());
      return tmp;
    }
  };

  class OneDimScaleOp : public LinOp {
   public:
    OneDimScaleOp() : LinOp("mv", 1, Type.Double, Device.cpu) {}
    UniTensor matvec_impl(const UniTensor& A) override { return A * 3.0; }
  };

  class BlockScaleOp : public LinOp {
   public:
    BlockScaleOp() : LinOp("mv", 0, Type.Double, Device.cpu) {}
    UniTensor matvec_impl(const UniTensor& A) override { return A * 3.0; }
  };

  class SmallResidualOp : public LinOp {
   public:
    explicit SmallResidualOp(const double coupling, const unsigned int dtype = Type.Double)
        : LinOp("mv", 3, dtype, Device.cpu), coupling_(coupling) {}

    UniTensor matvec_impl(const UniTensor& A) override {
      input_dtypes.push_back(A.dtype());
      auto out = UniTensor::zeros(A.shape(), A.labels(), A.dtype(), A.device());
      out.set_rowrank_(A.rowrank());
      out.at({0, 0}) = coupling_ * A.at({1, 0});
      out.at({1, 0}) = coupling_ * A.at({0, 0});
      return out;
    }

    std::vector<unsigned int> input_dtypes;

   private:
    double coupling_;
  };

  class TwoDimMixingOp : public LinOp {
   public:
    TwoDimMixingOp() : LinOp("mv", 2, Type.Double, Device.cpu) {}

    UniTensor matvec_impl(const UniTensor& A) override {
      auto out = UniTensor::zeros(A.shape(), A.labels(), A.dtype(), A.device());
      out.set_rowrank_(A.rowrank());
      out.at({0, 0}) = A.at({1, 0});
      out.at({1, 0}) = A.at({0, 0});
      return out;
    }
  };

  class ThreeDimChainOp : public LinOp {
   public:
    ThreeDimChainOp(const double coupling01, const double coupling12)
        : LinOp("mv", 3, Type.Double, Device.cpu),
          coupling01_(coupling01),
          coupling12_(coupling12) {}

    UniTensor matvec_impl(const UniTensor& A) override {
      auto out = UniTensor::zeros(A.shape(), A.labels(), A.dtype(), A.device());
      out.set_rowrank_(A.rowrank());
      out.at({0, 0}) = coupling01_ * A.at({1, 0});
      out.at({1, 0}) = coupling01_ * A.at({0, 0}) + coupling12_ * A.at({2, 0});
      out.at({2, 0}) = coupling12_ * A.at({1, 0});
      return out;
    }

    Tensor dense_hamiltonian() const {
      auto H = zeros({3, 3}, Type.Double);
      H.at({0, 1}) = coupling01_;
      H.at({1, 0}) = coupling01_;
      H.at({1, 2}) = coupling12_;
      H.at({2, 1}) = coupling12_;
      return H;
    }

   private:
    double coupling01_;
    double coupling12_;
  };

  // A 3-site Hermitian chain (couples 0-1-2) whose Krylov space starting from e0 is the
  // full 3-dimensional space, exhausted at k=3. dtype-parametrised so it can exercise the
  // float convergence path.
  class ChainOp3 : public LinOp {
   public:
    ChainOp3(const double coupling01, const double coupling12,
             const unsigned int dtype = Type.Double)
        : LinOp("mv", 3, dtype, Device.cpu), coupling01_(coupling01), coupling12_(coupling12) {}

    UniTensor matvec_impl(const UniTensor& A) override {
      auto out = UniTensor::zeros(A.shape(), A.labels(), A.dtype(), A.device());
      out.set_rowrank_(A.rowrank());
      out.at({0, 0}) = coupling01_ * A.at({1, 0});
      out.at({1, 0}) = coupling01_ * A.at({0, 0}) + coupling12_ * A.at({2, 0});
      out.at({2, 0}) = coupling12_ * A.at({1, 0});
      return out;
    }

   private:
    double coupling01_;
    double coupling12_;
  };

  double FloatLanczosExpTolerance() { return 100.0 * std::numeric_limits<float>::epsilon(); }

  UniTensor SmallResidualInitialState(const unsigned int dtype) {
    UniTensor Tin = UniTensor::zeros({3, 1}, {}, dtype, Device.cpu).set_rowrank_(1);
    Tin.at({0, 0}) = 1.0;
    return Tin;
  }

  UniTensor SmallResidualExpectedState(const double coupling, const double tau,
                                       const unsigned int dtype) {
    auto ans = UniTensor::zeros({3, 1}, {}, dtype, Device.cpu).set_rowrank_(1);
    ans.at({0, 0}) = std::cosh(coupling * tau);
    ans.at({1, 0}) = std::sinh(coupling * tau);
    return ans;
  }

  UniTensor SmallResidualExpectedState(const double coupling, const cytnx_complex128 tau,
                                       const unsigned int dtype) {
    auto ans = UniTensor::zeros({3, 1}, {}, dtype, Device.cpu).set_rowrank_(1);
    ans.at({0, 0}) = std::cosh(coupling * tau);
    ans.at({1, 0}) = std::sinh(coupling * tau);
    return ans;
  }

  UniTensor TwoDimInitialState() {
    auto Tin = UniTensor::zeros({2, 1}, {}, Type.Double, Device.cpu).set_rowrank_(1);
    Tin.at({0, 0}) = 1.0;
    return Tin;
  }

  UniTensor TwoDimExpectedState(const double tau) {
    auto ans = UniTensor::zeros({2, 1}, {}, Type.Double, Device.cpu).set_rowrank_(1);
    ans.at({0, 0}) = std::cosh(tau);
    ans.at({1, 0}) = std::sinh(tau);
    return ans;
  }

  UniTensor ThreeDimInitialState(const double scale = 1.0) {
    auto Tin = UniTensor::zeros({3, 1}, {}, Type.Double, Device.cpu).set_rowrank_(1);
    Tin.at({0, 0}) = scale;
    return Tin;
  }

  UniTensor BlockInitialState() {
    Bond lan_I = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {2, 2, 2});
    Bond lan_J = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {1, 1, 1});
    UniTensor state({lan_I, lan_J});
    state.relabel_({"in", "out"});
    state.put_block(arange(2).astype(Type.Double).reshape({2, 1}) + 1.0, 0);
    state.put_block(arange(2).astype(Type.Double).reshape({2, 1}) + 3.0, 1);
    state.put_block(arange(2).astype(Type.Double).reshape({2, 1}) + 5.0, 2);
    return state;
  }

  UniTensor DenseExpectedState(const ThreeDimChainOp& op, const UniTensor& Tin, const double tau) {
    auto ans = linalg::Matmul(linalg::ExpH(op.dense_hamiltonian(), tau), Tin.get_block());
    return UniTensor(ans, false, Tin.rowrank());
  }

  // describe:Real type test
  TEST(Lanczos_Exp_Ut, RealTypeTest) {
    int d = 2, D = 5;
    auto op = OneSiteOp(d, D);
    auto Tin = CreateA(d, D);
    const double crit = 1.0e-8;
    double tau = 0.1;
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
    auto ans = GetAns(op.EffH, Tin, tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());
    EXPECT_EQ(x.dtype(), Type.Double);
    EXPECT_LE(err, crit);
  }

  // describe:Complex type test
  TEST(Lanczos_Exp_Ut, ComplexTypeTest) {
    int d = 2, D = 5;
    auto op = OneSiteOp(d, D, Type.ComplexDouble);
    auto Tin = CreateA(d, D, Type.ComplexDouble);
    const double crit = 1.0e-8;
    std::complex<double> tau = std::complex<double>(0, 1) * 0.1;
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
    auto ans = GetAns(op.EffH, Tin, tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());
    EXPECT_LE(err, crit);
  }

  // describe:Test non-Hermitian Op but the code will not crash
  TEST(Lanczos_Exp_Ut, NonHermit) {
    int d = 2, D = 5;
    double low = -1.0, high = 1.0;
    auto op = OneSiteOp(d, D);
    op.EffH.uniform_(low, high, 0);
    auto Tin = CreateA(d, D);
    const double crit = 1.0e-3;
    double tau = 0.1;
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
  }

  // describe:input |v| != 1
  TEST(Lanczos_Exp_Ut, normVNot1) {
    int d = 2, D = 5;
    auto op = OneSiteOp(d, D);
    auto Tin = CreateA(d, D) * 1.1;
    const double crit = 1.0e-7;
    double tau = 0.1;
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
    auto ans = GetAns(op.EffH, Tin, tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());
    EXPECT_LE(err, crit);
  }

  TEST(Lanczos_Exp_Ut, OneDimensionalKrylovSpace) {
    OneDimScaleOp op;
    UniTensor Tin = UniTensor::zeros({1, 1}, {}, Type.Double, Device.cpu).set_rowrank_(1);
    Tin.at({0, 0}) = 2.0;
    const double crit = 1.0e-8;
    const double tau = 0.2;
    linalg::clear_krylov_stats();

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
    auto ans = Tin * std::exp(3.0 * tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_LE(err, crit);
    auto stats = linalg::last_krylov_stats();
    EXPECT_EQ(stats.algorithm, "Lanczos_Exp");
    EXPECT_TRUE(stats.converged);
    EXPECT_EQ(stats.reason, "full_krylov_dimension");
    EXPECT_EQ(stats.krylov_dim, 1);
    EXPECT_EQ(stats.matvec_count, 1);
    EXPECT_EQ(stats.input_dtype, Type.Double);
    EXPECT_EQ(stats.op_dtype, Type.Void);
    EXPECT_EQ(stats.working_dtype, Type.Double);
    auto total_stats = linalg::krylov_stats();
    EXPECT_EQ(total_stats.matvec_count, stats.matvec_count);
    EXPECT_EQ(total_stats.krylov_dim, stats.krylov_dim);
    EXPECT_EQ(total_stats.op_dtype, stats.op_dtype);
  }

  TEST(Lanczos_Exp_Ut, BlockUniTensorOneDimensionalKrylovSpace) {
    BlockScaleOp op;
    auto Tin = BlockInitialState();
    const double crit = 1.0e-8;
    const double tau = 0.2;

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit);
    auto ans = Tin * std::exp(3.0 * tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_EQ(x.uten_type(), UTenType.Block);
    EXPECT_EQ(x.labels(), Tin.labels());
    EXPECT_EQ(x.shape(), Tin.shape());
    EXPECT_EQ(x.rowrank(), Tin.rowrank());
    EXPECT_LE(err, crit);
  }

  TEST(Lanczos_Exp_Ut, FullKrylovSpaceDoesNotWarnAtDimensionLimit) {
    TwoDimMixingOp op;
    auto Tin = TwoDimInitialState();
    const double crit = 1.0e-8;
    const double tau = 1.0;
    const unsigned int maxiter = 2;

    testing::internal::CaptureStderr();
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit, maxiter);
    const std::string stderr_output = testing::internal::GetCapturedStderr();
    auto ans = TwoDimExpectedState(tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_EQ(stderr_output.find("[WARNING][Lanczos_Exp]"), std::string::npos) << stderr_output;
    EXPECT_LE(err, crit);
  }

  TEST(Lanczos_Exp_Ut, UsesBetaWeightedExponentialErrorEstimate) {
    ThreeDimChainOp op(1.0, 1.0);
    auto Tin = ThreeDimInitialState();
    const double crit = 1.0e-8;
    const double tau = 1.0e-5;
    const unsigned int maxiter = 2;

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit, maxiter);
    auto ans = DenseExpectedState(op, Tin, tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_LE(err, crit);
    auto stats = linalg::last_krylov_stats();
    EXPECT_EQ(stats.reason, "projected_exponential");
    EXPECT_EQ(stats.krylov_dim, 2);
    EXPECT_LT(stats.final_error, crit);
  }

  TEST(Lanczos_Exp_Ut, ConvergenceCriterionScalesWithInputNorm) {
    ThreeDimChainOp op(1.0, 1.0);
    auto Tin = ThreeDimInitialState(1.0e6);
    const double crit = 1.0e-8;
    const double tau = 1.0e-5;
    const unsigned int maxiter = 3;

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit, maxiter);
    auto ans = DenseExpectedState(op, Tin, tau);
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_LE(err, crit);
    auto stats = linalg::last_krylov_stats();
    EXPECT_EQ(stats.reason, "breakdown");
    EXPECT_EQ(stats.krylov_dim, 3);
  }

  TEST(Lanczos_Exp_Ut, SmallResidualIsNotBreakdown) {
    const double coupling = 5.0e-7;
    SmallResidualOp op(coupling);
    auto Tin = SmallResidualInitialState(Type.Double);
    const double crit = 1.0e-8;
    const double tau = 1.0;
    const unsigned int maxiter = 3;

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, crit, maxiter);
    auto ans = SmallResidualExpectedState(coupling, tau, Type.Double);
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_LE(err, crit);
  }

  TEST(Lanczos_Exp_Ut, FloatSmallResidualBelowRoundoffFloor) {
    const double coupling = 5.0e-6;
    SmallResidualOp op(coupling, Type.Float);
    auto Tin = SmallResidualInitialState(Type.Float);
    const double tau = 1.0;
    const unsigned int maxiter = 3;

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-10, maxiter);
    auto ans = SmallResidualExpectedState(coupling, tau, x.dtype());
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_EQ(x.dtype(), Type.Float);
    EXPECT_LE(err, FloatLanczosExpTolerance());
  }

  TEST(Lanczos_Exp_Ut, FloatResidualAboveRoundoffFloorIsNotBreakdown) {
    const double coupling = 5.0e-5;
    SmallResidualOp op(coupling, Type.Float);
    auto Tin = SmallResidualInitialState(Type.Float);
    const double tau = 1.0;
    const unsigned int maxiter = 3;

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-10, maxiter);
    auto ans = SmallResidualExpectedState(coupling, tau, x.dtype());
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_EQ(x.dtype(), Type.Float);
    EXPECT_LE(err, FloatLanczosExpTolerance());
  }

  TEST(Lanczos_Exp_Ut, ComplexFloatResidualAboveRoundoffFloorIsNotBreakdown) {
    const double coupling = 5.0e-5;
    SmallResidualOp op(coupling, Type.ComplexFloat);
    auto Tin = SmallResidualInitialState(Type.ComplexFloat);
    const double tau = 1.0;
    const unsigned int maxiter = 3;

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-10, maxiter);
    auto ans = SmallResidualExpectedState(coupling, tau, Type.ComplexFloat);
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_EQ(x.dtype(), Type.ComplexFloat);
    EXPECT_LE(err, FloatLanczosExpTolerance());
  }

  // A requested CvgCrit below the dtype-dependent orthogonality floor (~sqrt(eps)) must be
  // clamped up to that floor, because a non-reorthogonalized Lanczos basis cannot deliver
  // accuracy below sqrt(eps). The clamped value is reported through last_krylov_stats().
  // Maxiter (2) < nx (3) so the Krylov space is not exhaustible within the budget and the
  // floor genuinely applies (see FloatFullKrylovSpaceReachedDespiteFloor for the
  // exhaustible case, where the floor must NOT clamp).
  TEST(Lanczos_Exp_Ut, ConvergenceCriterionClampedToOrthogonalityFloor) {
    const double tiny_request = 1.0e-30;
    const double tau = 0.1;
    const unsigned int maxiter = 2;

    // Double: floor is kept at the round 1e-8 (~ sqrt(double eps) = 1.49e-8).
    {
      SmallResidualOp op(5.0e-5, Type.Double);
      auto Tin = SmallResidualInitialState(Type.Double);
      linalg::clear_krylov_stats();
      linalg::Lanczos_Exp(&op, Tin, tau, tiny_request, maxiter);
      auto stats = linalg::last_krylov_stats();
      EXPECT_DOUBLE_EQ(stats.cvgcrit_requested, tiny_request);
      EXPECT_DOUBLE_EQ(stats.cvgcrit_used, 1.0e-8);
    }

    // Float: floor is sqrt(float eps) ~ 3.45e-4, not the old roundoff scale 100*eps.
    {
      SmallResidualOp op(5.0e-5, Type.Float);
      auto Tin = SmallResidualInitialState(Type.Float);
      linalg::clear_krylov_stats();
      linalg::Lanczos_Exp(&op, Tin, tau, tiny_request, maxiter);
      auto stats = linalg::last_krylov_stats();
      const double expected_floor =
        std::sqrt(static_cast<double>(std::numeric_limits<float>::epsilon()));
      EXPECT_DOUBLE_EQ(stats.cvgcrit_used, expected_floor);
      EXPECT_GT(stats.cvgcrit_used, 100.0 * std::numeric_limits<float>::epsilon());
    }
  }

  // Regression (codex review of the sqrt(eps) float floor): the floor must not stop a solve
  // before the Krylov space is exhausted when the full space is still reachable. The 3-site
  // chain from e0 reaches the whole space at k=3; with tau=0.02 the k=2 error bound
  // ~ tau^2/2 = 2e-4 sits below the float floor (3.45e-4), so a naive clamp would converge
  // at k=2 and omit the ~2e-4 third component instead of continuing to the exact answer.
  TEST(Lanczos_Exp_Ut, FloatFullKrylovSpaceReachedDespiteFloor) {
    const double c = 1.0;
    const double tau = 0.02;
    const unsigned int maxiter = 3;  // == nx: the full Krylov space is exhaustible
    ChainOp3 op(c, c, Type.Float);
    auto Tin = SmallResidualInitialState(Type.Float);  // e0 = [1,0,0]

    linalg::clear_krylov_stats();
    auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-8, maxiter);
    auto stats = linalg::last_krylov_stats();

    // Reference: dense float matrix exponential of the same operator applied to e0.
    auto H = zeros({3, 3}, Type.Float);
    H.at<cytnx_float>({0, 1}) = c;
    H.at<cytnx_float>({1, 0}) = c;
    H.at<cytnx_float>({1, 2}) = c;
    H.at<cytnx_float>({2, 1}) = c;
    auto ref =
      UniTensor(linalg::Matmul(linalg::ExpH(H, tau), Tin.get_block()), false, Tin.rowrank());
    const auto err = static_cast<double>((x - ref).Norm().item().real());

    EXPECT_EQ(x.dtype(), Type.Float);
    EXPECT_EQ(stats.krylov_dim, 3u);  // full space reached, not truncated at k=2
    EXPECT_LT(err, 5.0e-5);           // ~float precision, not the ~2e-4 truncation error
  }

  TEST(Lanczos_Exp_Ut, FloatComplexTauReturnsComplexFloat) {
    const double coupling = 5.0e-5;
    SmallResidualOp op(coupling, Type.Float);
    auto Tin = SmallResidualInitialState(Type.Float);
    const cytnx_complex128 tau(0.0, 1.0);
    const unsigned int maxiter = 3;

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-10, maxiter);
    auto ans = SmallResidualExpectedState(coupling, tau, Type.ComplexFloat);
    auto err = static_cast<double>((x - ans).Norm().item().real());

    EXPECT_EQ(x.dtype(), Type.ComplexFloat);
    ASSERT_FALSE(op.input_dtypes.empty());
    for (const auto dtype : op.input_dtypes) {
      EXPECT_EQ(dtype, Type.Float);
    }
    EXPECT_LE(err, FloatLanczosExpTolerance());
  }

  TEST(Lanczos_Exp_Ut, IgnoresLinOpDTypeHint) {
    const double coupling = 5.0e-5;
    SmallResidualOp op(coupling, Type.Float);
    auto Tin = SmallResidualInitialState(Type.Double);
    const double tau = 1.0;
    const unsigned int maxiter = 3;

    auto x = linalg::Lanczos_Exp(&op, Tin, tau, 1.0e-10, maxiter);

    EXPECT_EQ(x.dtype(), Type.Double);
    auto stats = linalg::last_krylov_stats();
    EXPECT_EQ(stats.op_dtype, Type.Void);
    EXPECT_EQ(stats.working_dtype, Type.Double);
  }

  // describe:test incorrect data type
  TEST(Lanczos_Exp_Ut, IncorrectDType) {
    int d = 2, D = 10;
    auto op = OneSiteOp(d, D, Type.Int64);
    auto Tin = CreateA(d, D, Type.Int64);
    const double crit = 1.0e-3;
    double tau = 0.1;
    EXPECT_THROW({ linalg::Lanczos_Exp(&op, Tin, crit, tau); }, std::logic_error);
  }

  // describe:test not supported UniTensor Type

  /*
   *     -1
   *     |
   *  0--A--2
   */
  UniTensor CreateA(const int d, const int D, const unsigned int dtype, const int device) {
    double low = -1.0, high = 1.0;
    UniTensor A = UniTensor({Bond(D), Bond(d), Bond(D)}, {}, -1, dtype, device)
                    .set_name("A")
                    .relabel_({"vol", "po", "vor"})
                    .set_rowrank_(1);
    if (Type.is_float(A.dtype())) {
      random::uniform_(A, low, high, 0);
    }
    // A = A / std::sqrt(double((Dot(A, A).real())));
    return A;
  }

  /*
   *         |-|--"vil" "pi" "vir"--|-|
   *         | |         +          | |
   *         |L|- -------O----------|R|
   *         | |         +          | |
   *         |_|--"vol" "po" "vor"--|_|
   */
  UniTensor CreateOneSiteEffHam(const int d, const int D, const unsigned int dtype,
                                const int device) {
    double low = -1.0, high = 1.0;
    std::vector<Bond> bonds = {Bond(D), Bond(d), Bond(D), Bond(D), Bond(d), Bond(D)};
    std::vector<std::string> heff_labels = {"vil", "pi", "vir", "vol", "po", "vor"};
    UniTensor HEff = UniTensor(bonds, {}, -1, dtype, device)
                       .set_name("HEff")
                       .relabel_(heff_labels)
                       .set_rowrank(bonds.size() / 2);
    auto HEff_shape = HEff.shape();
    auto in_dim = 1;
    for (int i = 0; i < HEff.rowrank(); ++i) {
      in_dim *= HEff_shape[i];
    }
    auto out_dim = in_dim;
    if (Type.is_float(HEff.dtype())) {
      random::uniform_(HEff, low, high, 0);
    }
    auto HEff_mat = HEff.get_block();
    HEff_mat.reshape_({in_dim, out_dim});
    HEff_mat = HEff_mat + HEff_mat.permute({1, 0});  // symmtrize

    // Let H can be converge in ExpM
    auto eigs = HEff_mat.Eigh();
    auto e = UniTensor(eigs[0], true) * 0.01;
    e.relabel_({"a", "b"});
    auto v = UniTensor(eigs[1]);
    v.relabel_({"i", "a"});
    auto vt = UniTensor(linalg::InvM(v.get_block()));
    vt.relabel_({"b", "j"});
    HEff_mat = Contract(Contract(e, v), vt).get_block();

    // HEff_mat = linalg::Matmul(HEff_mat, HEff_mat.permute({1, 0}).Conj());  // positive definete
    HEff_mat.reshape_(HEff_shape);
    HEff.put_block(HEff_mat);
    return HEff;
  }

  UniTensor GetAns(const UniTensor& HEff, const UniTensor& Tin, const Scalar& tau) {
    auto expH = HEff.clone();
    auto HEff_shape = HEff.shape();
    auto in_dim = 1;
    for (int i = 0; i < HEff.rowrank(); ++i) {
      in_dim *= HEff_shape[i];
    }
    auto out_dim = in_dim;
    // we use ExpM since tau*H will not be Hermitian if tau is complex number even H is Hermitian
    expH.put_block(
      linalg::ExpM((tau * expH.get_block()).reshape(in_dim, out_dim)).reshape(HEff_shape));
    auto ans = Contract(expH, Tin);
    ans.permute_({"vil", "pi", "vir"}, 1);
    ans.relabel_(Tin.labels());
    ans = Contract(expH, Tin);
    return ans;
  }

}  // namespace Lanczos_Exp_Ut_Test
