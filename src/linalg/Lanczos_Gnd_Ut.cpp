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
#include "UniTensor.hpp"
#include "utils/vec_print.hpp"

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

      unsigned int hermitian_projection_dtype_internal(const unsigned int dtype) {
        return dtype < 3 ? dtype + 2 : dtype;
      }

      Scalar real_scalar_for_dtype_internal(const double value, const unsigned int dtype) {
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          return Scalar(static_cast<cytnx_float>(value));
        }
        return Scalar(static_cast<cytnx_double>(value));
      }

      double scalar_abs_internal(const Scalar &s) { return static_cast<double>(s.abs()); }

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

    // <A|B>
    static Scalar _Dot(const UniTensor &A, const UniTensor &B) {
      // For a fermionic tensor the bra-ket contraction carries sign flips; applying
      // fermion_twists() to the dagger (bra) operand cancels them so the result is the true scalar
      // product. For bosonic/dense tensors fermion_twists() is not defined, so use the plain
      // contraction.
      if (A.uten_type() == UTenType.BlockFermionic)
        return Contract(A.Dagger().fermion_twists(), B).item();
      return Contract(A.Dagger(), B).item();
    }

    void _Lanczos_Gnd_general_Ut(std::vector<UniTensor> &out, LinOp *Hop, const UniTensor &Tin,
                                 bool is_V, double CvgCrit, unsigned int Maxiter, bool verbose,
                                 KrylovStats *stats) {
      out.clear();
      std::vector<UniTensor> psi_s;

      // Frobenius norm is sign-flip independent, so it is the correct vector norm for fermionic
      // tensors too (and equals sqrt(<Tin|Tin>) for bosonic).
      const unsigned int input_dtype = Tin.dtype();
      const double eps = dtype_epsilon_internal(input_dtype);
      const std::uint64_t vec_len = Hop->nx();
      const unsigned int imp_maxiter = capped_nonrestarted_maxiter_internal(Maxiter, vec_len);
      if (stats) {
        stats->maxiter_used = imp_maxiter;
      }
      auto Norm = real_scalar_for_dtype_internal(double(Tin.Norm().item().real()), input_dtype);
      cytnx_error_msg(double(Norm.real()) == 0.0,
                      "[ERROR][Lanczos_Gnd] initial vector has zero norm.%s", "\n");

      UniTensor psi_1 = Tin / Norm;
      psi_1.contiguous_();
      // Pin the Krylov vectors to the sign frame with all signflips applied and the vector
      // arithmetic below stay consistent; no-op for bosonic/dense tensors.
      psi_1.apply_();
      if (is_V) {
        psi_s.push_back(psi_1);
      }

      UniTensor psi_0;
      UniTensor new_psi;
      bool cvg_fin = false;

      Tensor As = zeros({1}, hermitian_projection_dtype_internal(Tin.dtype()), Tin.device());
      Tensor Bs = As.clone();
      Scalar E;

      std::vector<Tensor> tmpEsVs;

      new_psi = Hop->matvec(psi_1);
      if (stats) {
        stats->matvec_count++;
      }
      new_psi.apply_();  // resolve pending fermionic signflip; no-op for bosonic/dense

      cytnx_error_msg(new_psi.labels().size() != psi_1.labels().size(),
                      "[ERROR] LinOp.matvec(UniTensor) -> UniTensor the output should have same "
                      "labels and shape as input!%s",
                      "\n");
      cytnx_error_msg(new_psi.labels() != psi_1.labels(),
                      "[ERROR] LinOp.matvec(UniTensor) -> UniTensor the output should have same "
                      "labels and shape as input!%s",
                      "\n");

      auto alpha = _Dot(new_psi, psi_1).real();
      As(0) = alpha;
      new_psi -= alpha * psi_1;
      auto beta = real_scalar_for_dtype_internal(double(new_psi.Norm().item().real()), input_dtype);
      Bs(0) = beta;

      auto beta_breakdown_scale = std::max(scalar_abs_internal(alpha), 1.0);
      auto beta_breakdown = kBetaBreakdownRoundoff * eps * beta_breakdown_scale;
      try {
        tmpEsVs = linalg::Tridiag(As, Bs, true, true, true);
      } catch (std::logic_error le) {
        std::cout << "[WARNING] Lanczos_Gnd -> Tridiag error: \n";
        std::cout << le.what() << std::endl;
        std::cout << "Lanczos stops automatically." << std::endl;
        if (stats) {
          stats->converged = false;
          stats->reason = "tridiag_error";
          stats->final_beta = scalar_abs_internal(beta);
          stats->breakdown_tol = beta_breakdown;
        }
        return;
      }
      if (scalar_abs_internal(beta) <= beta_breakdown || imp_maxiter == 1) {
        cvg_fin = true;
        if (stats) {
          stats->converged = true;
          stats->reason =
            scalar_abs_internal(beta) <= beta_breakdown ? "breakdown" : "full_krylov_dimension";
        }
      } else {
        psi_0 = psi_1;
        new_psi /= beta;
        psi_1 = new_psi;
        if (is_V) {
          psi_s.push_back(psi_1);
        }
      }
      E = tmpEsVs[0].storage().at(0);
      Scalar Ediff;

      ///---------------------------

      // iteration LZ:
      for (unsigned int i = 1; !cvg_fin && i < imp_maxiter; i++) {
        new_psi = Hop->matvec(psi_1);
        if (stats) {
          stats->matvec_count++;
        }
        new_psi.apply_();  // resolve pending fermionic signflip; no-op for bosonic/dense
        alpha = _Dot(new_psi, psi_1).real();
        As.append(alpha);
        new_psi -= (alpha * psi_1 + beta * psi_0);

        try {
          auto tmptmp = linalg::Tridiag(As, Bs, true, true, true);
          tmpEsVs = tmptmp;
        } catch (std::logic_error le) {
          std::cout << "[WARNING] Lanczos_Gnd -> Tridiag error: \n";
          std::cout << le.what() << std::endl;
          std::cout << "Lanczos stops automatically." << std::endl;
          if (stats) {
            stats->converged = false;
            stats->reason = "tridiag_error";
          }
          break;
        }

        beta = real_scalar_for_dtype_internal(double(new_psi.Norm().item().real()), input_dtype);
        Bs.append(beta);

        beta_breakdown_scale = std::max(
          beta_breakdown_scale,
          std::max(scalar_abs_internal(alpha) + scalar_abs_internal(Bs(i - 1).item()), 1.0));
        beta_breakdown =
          kBetaBreakdownRoundoff * eps * beta_breakdown_scale * std::sqrt(static_cast<double>(i));
        if (stats) {
          stats->final_beta = scalar_abs_internal(beta);
          stats->breakdown_tol = beta_breakdown;
        }
        if (scalar_abs_internal(beta) <= beta_breakdown) {
          cvg_fin = true;
          if (stats) {
            stats->converged = true;
            stats->reason = "breakdown";
          }
          break;
        }

        psi_0 = psi_1;
        psi_1 = new_psi / beta;
        if (is_V) {
          psi_s.push_back(psi_1);
        }
        Ediff = abs(E - tmpEsVs[0].storage().at(0));
        if (verbose) {
          printf("iter[%d] Enr: %11.14f, diff from last iter: %11.14f\n", i, double(E),
                 double(Ediff));
        }

        if (stats) {
          stats->final_error = double(Ediff);
        }
        if (Ediff < CvgCrit) {
          cvg_fin = true;
          if (stats) {
            stats->converged = true;
            stats->reason = "energy_diff";
          }
          break;
        }
        if (i == imp_maxiter - 1) {
          if (stats) {
            stats->converged = false;
            stats->reason = Maxiter < vec_len ? "maxiter" : "full_krylov_dimension";
          }
          break;
        }
        E = tmpEsVs[0].storage().at(0);

      }  // iteration

      if (stats) {
        stats->iterations = tmpEsVs[0].shape()[0];
        stats->krylov_dim = tmpEsVs[0].shape()[0];
        stats->final_beta = scalar_abs_internal(beta);
        stats->breakdown_tol = beta_breakdown;
      }

      out.push_back(UniTensor(tmpEsVs[0](0), false, 0));

      if (is_V) {
        UniTensor eV;
        Storage kryVg = tmpEsVs[1](0).storage();
        tmpEsVs.pop_back();

        eV = kryVg.at(0) * psi_s.at(0);
        for (unsigned int n = 1; n < tmpEsVs[0].shape()[0]; n++) {
          eV += kryVg.at(n) * psi_s.at(n);
        }

        out.push_back(eV);
      }
    }

    std::vector<UniTensor> Lanczos_Gnd_Ut(LinOp *Hop, const UniTensor &Tin, double CvgCrit,
                                          bool is_V, bool verbose, unsigned int Maxiter) {
      cytnx_error_msg(CvgCrit <= 0, "[ERROR][Lanczos] converge criteria must >0%s", "\n");
      cytnx_error_msg(Maxiter < 2, "[ERROR][Lanczos] Maxiter must >1%s", "\n");

      UniTensor v0;
      cytnx_error_msg(!Type.is_float(Tin.dtype()),
                      "[ERROR][Lanczos] Lanczos can only accept input tensors with floating types "
                      "(complex/real)%s",
                      "\n");
      v0 = Tin.astype(promoted_working_dtype_internal(Tin.dtype(), Hop->dtype()));

      std::vector<UniTensor> out;

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

      KrylovStats stats;
      stats.algorithm = "Lanczos_Gnd_Ut";
      stats.maxiter_requested = Maxiter;
      stats.cvgcrit_requested = CvgCrit;
      stats.cvgcrit_used = _cvgcrit;
      stats.input_dtype = Tin.dtype();
      stats.working_dtype = v0.dtype();

      _Lanczos_Gnd_general_Ut(out, Hop, v0, is_V, _cvgcrit, Maxiter, verbose, &stats);
      set_last_krylov_stats(stats);

      return out;

    }  // Lanczos_Gnd entry point

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
