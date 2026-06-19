#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;
    using namespace std;

    namespace {
      constexpr double kBetaBreakdownRoundoff = 100.0;

      unsigned int promoted_working_dtype_internal(const unsigned int input_dtype,
                                                   const unsigned int op_dtype) {
        if (input_dtype == Type.Void) {
          return op_dtype == Type.Void ? Type.Double : op_dtype;
        }
        if (op_dtype == Type.Void) {
          return input_dtype;
        }
        return Type.type_promote(input_dtype, op_dtype);
      }

      double dtype_epsilon_internal(const unsigned int dtype) {
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          return std::numeric_limits<float>::epsilon();
        }
        return std::numeric_limits<double>::epsilon();
      }

      double float_cvgcrit_floor_internal() {
        return kBetaBreakdownRoundoff * std::numeric_limits<float>::epsilon();
      }

      double scalar_abs_internal(const Scalar &s) { return static_cast<double>(s.abs()); }

      Scalar real_scalar_for_dtype_internal(const double value, const unsigned int dtype) {
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          return Scalar(static_cast<cytnx_float>(value));
        }
        return Scalar(static_cast<cytnx_double>(value));
      }

      unsigned int capped_nonrestarted_maxiter_internal(const unsigned int maxiter,
                                                        const std::uint64_t vec_len) {
        constexpr unsigned int kMaxNonrestartedMaxiter = 100;
        const unsigned int finite_space_cap =
          vec_len < maxiter ? static_cast<unsigned int>(vec_len) : maxiter;
        static bool warning_issued = false;
        if (finite_space_cap > kMaxNonrestartedMaxiter && !warning_issued) {
          warning_issued = true;
          cytnx_warning_msg(
            true,
            "[WARNING][Lanczos_Gnd] Non-restarted Lanczos would use %u Krylov steps after the "
            "finite-space cap, which is unusually large. Capping Maxiter to %u. Use the "
            "ARPACK-backed Lanczos(..., which=\"SA\") entry point for generic ground-state "
            "eigensolver use.%s",
            finite_space_cap, kMaxNonrestartedMaxiter, "\n");
        }
        return std::min(finite_space_cap, kMaxNonrestartedMaxiter);
      }
    }  // namespace

    void _Lanczos_Gnd_general(std::vector<Tensor> &out, LinOp *Hop, const Tensor &Tin, bool is_V,
                              double CvgCrit, unsigned int Maxiter, bool verbose) {
      out.clear();

      const unsigned int input_dtype = Tin.dtype();
      const double eps = dtype_epsilon_internal(input_dtype);
      const std::uint64_t vec_len = Hop->nx();
      const unsigned int imp_maxiter = capped_nonrestarted_maxiter_internal(Maxiter, vec_len);

      const double tin_norm = static_cast<double>(Tin.Norm().item().real());
      cytnx_error_msg(tin_norm == 0.0, "[ERROR][Lanczos_Gnd] initial vector has zero norm.%s",
                      "\n");
      Tensor psi_1 = Tin.clone() / real_scalar_for_dtype_internal(tin_norm, input_dtype);

      Tensor psi_0;
      Tensor new_psi;
      bool cvg_fin = false;

      Tensor As = zeros({1}, Tin.dtype(), Tin.device());
      Tensor Bs = As.clone();

      Scalar E = Scalar::maxval(Tin.dtype());
      std::vector<Tensor> tmpEsVs;

      new_psi = Hop->matvec(psi_1);

      Scalar alpha = linalg::Vectordot(new_psi, psi_1, true).item();
      As(0) = alpha;
      new_psi -= alpha * psi_1;

      double beta = static_cast<double>(new_psi.Norm().item().real());
      Bs(0) = real_scalar_for_dtype_internal(beta, input_dtype);

      auto beta_breakdown_scale = std::max(scalar_abs_internal(alpha), 1.0);
      auto beta_breakdown = kBetaBreakdownRoundoff * eps * beta_breakdown_scale;

      try {
        tmpEsVs = linalg::Tridiag(As, Bs, true, true, true);
      } catch (std::logic_error le) {
        std::cout << "[WARNING] Lanczos_Gnd -> Tridiag error: \n";
        std::cout << le.what() << std::endl;
        std::cout << "Lanczos stops automatically." << std::endl;
        return;
      }

      if (beta <= beta_breakdown || imp_maxiter == 1) {
        cvg_fin = true;
      } else {
        psi_0 = psi_1;
        new_psi /= Bs(0);
        psi_1 = new_psi;
      }

      E = tmpEsVs[0].storage().at(0);
      Scalar Ediff;

      ///---------------------------

      // iteration LZ:
      for (unsigned int i = 1; !cvg_fin && i < imp_maxiter; i++) {
        new_psi = Hop->matvec(psi_1);

        alpha = linalg::Vectordot(new_psi, psi_1, true).item();
        As.append(alpha);

        new_psi -= As(i) * psi_1;
        new_psi -= Bs(i - 1) * psi_0;

        try {
          auto tmptmp = linalg::Tridiag(As, Bs, true, true, true);
          tmpEsVs = tmptmp;
        } catch (std::logic_error le) {
          std::cout << "[WARNING] Lanczos_Gnd -> Tridiag error: \n";
          std::cout << le.what() << std::endl;
          std::cout << "Lanczos stops automatically." << std::endl;
          break;
        }
        beta = static_cast<double>(new_psi.Norm().item().real());
        Bs.append(real_scalar_for_dtype_internal(beta, input_dtype));
        beta_breakdown_scale = std::max(
          beta_breakdown_scale,
          std::max(scalar_abs_internal(alpha) + scalar_abs_internal(Bs(i - 1).item()), 1.0));
        beta_breakdown =
          kBetaBreakdownRoundoff * eps * beta_breakdown_scale * std::sqrt(static_cast<double>(i));
        if (beta <= beta_breakdown) {
          cvg_fin = true;
          break;
        }

        psi_0 = psi_1;
        new_psi /= Bs(i);

        psi_1 = new_psi;

        Ediff = abs(E - tmpEsVs[0].storage().at(0));
        if (verbose)
          printf("iter[%d] Enr: %11.14f, diff from last iter: %11.14f\n", i, double(E),
                 double(Ediff));

        if (Ediff < CvgCrit) {
          cvg_fin = true;
          break;
        }
        if (i == imp_maxiter - 1) break;
        E = tmpEsVs[0].storage().at(0);

      }  // iteration

      out.push_back(tmpEsVs[0](0));

      if (is_V) {
        Tensor eV;
        Tensor kryVg = tmpEsVs[1](0);
        tmpEsVs.pop_back();

        // Reconstruct the eigenvector from the stored tridiagonal eigenvector coefficients.
        psi_1 = Tin.clone() / real_scalar_for_dtype_internal(tin_norm, input_dtype);
        eV = kryVg(0) * psi_1;
        if (tmpEsVs[0].shape()[0] > 1) {
          new_psi = Hop->matvec(psi_1) - As(0) * psi_1;
          psi_0 = psi_1;
          psi_1 = new_psi / Bs(0);
        }

        for (unsigned int n = 1; n < tmpEsVs[0].shape()[0]; n++) {
          eV += kryVg(n) * psi_1;
          new_psi = Hop->matvec(psi_1) - Bs(n - 1) * psi_0;
          new_psi -= As(n) * psi_1;

          psi_0 = psi_1;
          psi_1 = new_psi / Bs(n);
        }

        out.push_back(eV);
      }
    }
    std::vector<Tensor> Lanczos_Gnd(LinOp *Hop, double CvgCrit, bool is_V, const Tensor &Tin,
                                    bool verbose, unsigned int Maxiter) {
      cytnx_error_msg(CvgCrit <= 0, "[ERROR][Lanczos] converge criteria must >0%s", "\n");
      cytnx_error_msg(Maxiter < 2, "[ERROR][Lanczos] Maxiter must >1%s", "\n");

      Tensor v0;
      if (Tin.dtype() == Type.Void) {
        v0 = cytnx::random::normal({Hop->nx()}, 0, 1, Hop->device())
               .astype(promoted_working_dtype_internal(Type.Void, Hop->dtype()));
      } else {
        cytnx_error_msg(Tin.shape().size() != 1, "[ERROR][Lanczos] Tin should be rank-1%s", "\n");
        cytnx_error_msg(Tin.shape()[0] != Hop->nx(),
                        "[ERROR][Lanczos] Tin should have dimension consistent with Hop: [%d] %s",
                        Hop->nx(), "\n");
        cytnx_error_msg(
          !Type.is_float(Tin.dtype()),
          "[ERROR][Lanczos] Lanczos can only accept input tensors with floating types "
          "(complex/real)%s",
          "\n");
        v0 = Tin.astype(promoted_working_dtype_internal(Tin.dtype(), Hop->dtype()));
      }

      std::vector<Tensor> out;

      double _cvgcrit = CvgCrit;

      if (v0.dtype() == Type.Float || v0.dtype() == Type.ComplexFloat) {
        const double cvgcrit_floor = float_cvgcrit_floor_internal();
        if (_cvgcrit < cvgcrit_floor) {
          _cvgcrit = cvgcrit_floor;
          cytnx_warning_msg(
            true,
            "[WARNING][Lanczos_Gnd] for float precision type, CvgCrit cannot be smaller "
            "than %.8e, and is automatically raised to this value.%s",
            cvgcrit_floor, "\n");
        }
      }

      _Lanczos_Gnd_general(out, Hop, v0, is_V, _cvgcrit, Maxiter, verbose);

      return out;

    }  // Lanczos_Gnd entry point

  }  // namespace linalg
}  // namespace cytnx

#endif
