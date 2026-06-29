#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <vector>
#include "UniTensor.hpp"
#include "utils/vec_print.hpp"
#include <iomanip>

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;
    using namespace std;

    namespace {
      constexpr double kBetaBreakdownRoundoff = 100.0;

      // <A|B>
      Scalar Dot_internal(const UniTensor &A, const UniTensor &B) {
        if (A.uten_type() == UTenType.BlockFermionic) {
          return Contract(A.Dagger().fermion_twists(), B).item();
        }
        return Contract(A.Dagger(), B).item();
      }

      // project v to u
      UniTensor Gram_Schmidt_proj_internal(const UniTensor &v, const UniTensor &u) {
        auto nu = Dot_internal(u, v);
        auto de = Dot_internal(u, u);
        auto coe = nu / de;
        return coe * u;
      }

      UniTensor Gram_Schmidt_internal(const std::vector<UniTensor> &vs) {
        auto u = vs.at(0).clone();
        double low = -1.0, high = 1.0;
        random::uniform_(u, low, high);
        for (auto &v : vs) {
          u -= Gram_Schmidt_proj_internal(u, v);
        }
        return u;
      }

      double dtype_epsilon_internal(const unsigned int dtype) {
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          return std::numeric_limits<float>::epsilon();
        }
        return std::numeric_limits<double>::epsilon();
      }

      double cvgcrit_floor_internal(const unsigned int dtype) {
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          return kBetaBreakdownRoundoff * std::numeric_limits<float>::epsilon();
        }
        return 1.0e-8;
      }

      bool should_warn_cvgcrit_floor_internal(const unsigned int dtype) {
        static bool warned_float = false;
        static bool warned_complex_float = false;
        static bool warned_double = false;
        static bool warned_complex_double = false;

        bool *warned = nullptr;
        if (dtype == Type.Float) {
          warned = &warned_float;
        } else if (dtype == Type.ComplexFloat) {
          warned = &warned_complex_float;
        } else if (dtype == Type.Double) {
          warned = &warned_double;
        } else if (dtype == Type.ComplexDouble) {
          warned = &warned_complex_double;
        }
        if (warned == nullptr || *warned) {
          return false;
        }
        *warned = true;
        return true;
      }

      unsigned int krylov_matrix_dtype_internal(const unsigned int dtype) {
        if (dtype == Type.Float) {
          return Type.Double;
        }
        if (dtype == Type.ComplexFloat) {
          return Type.ComplexDouble;
        }
        return dtype;
      }

      unsigned int complex_dtype_for_real_dtype_internal(const unsigned int dtype) {
        if (dtype == Type.Float) {
          return Type.ComplexFloat;
        }
        if (dtype == Type.Double) {
          return Type.ComplexDouble;
        }
        return dtype;
      }

      cytnx_uint64 tensor_element_count_internal(const Tensor &tensor) {
        cytnx_uint64 count = 1;
        for (const auto dim : tensor.shape()) count *= dim;
        return count;
      }

      unsigned int lanczos_exp_output_dtype_internal(const unsigned int working_dtype,
                                                     const unsigned int tau_dtype) {
        if (Type.is_complex(working_dtype)) {
          return working_dtype;
        }
        if (Type.is_complex(tau_dtype)) {
          return complex_dtype_for_real_dtype_internal(working_dtype);
        }
        return working_dtype;
      }

      unsigned int promoted_output_dtype_internal(const unsigned int input_dtype,
                                                  const unsigned int tau_dtype) {
        return lanczos_exp_output_dtype_internal(input_dtype, tau_dtype);
      }

      unsigned int projected_exponential_output_dtype_internal(const unsigned int output_dtype,
                                                               const unsigned int projected_dtype) {
        return output_dtype == Type.Void ? projected_dtype : output_dtype;
      }

      void cast_dense_output_internal(UniTensor &out, const unsigned int dtype) {
        if (out.dtype() == dtype) {
          return;
        }
        out = out.astype(dtype);
      }

      void check_matvec_output_internal(const UniTensor &out, const UniTensor &reference) {
        cytnx_error_msg(out.labels().size() != reference.labels().size(),
                        "[ERROR][Lanczos_Exp] LinOp.matvec(UniTensor) output must have the same "
                        "labels and shape as input.%s",
                        "\n");
        cytnx_error_msg(out.labels() != reference.labels(),
                        "[ERROR][Lanczos_Exp] LinOp.matvec(UniTensor) output must have the same "
                        "labels as input.%s",
                        "\n");
        cytnx_error_msg(out.shape() != reference.shape(),
                        "[ERROR][Lanczos_Exp] LinOp.matvec(UniTensor) output must have the same "
                        "shape as input.%s",
                        "\n");
      }

      UniTensor reconstruct_from_krylov_basis_internal(const std::vector<UniTensor> &basis,
                                                       const Tensor &projected_exponential,
                                                       const double input_norm) {
        cytnx_error_msg(basis.empty(), "[ERROR][Lanczos_Exp] Krylov basis is empty.%s", "\n");
        const auto krylov_dim = static_cast<cytnx_uint64>(basis.size());
        cytnx_error_msg(
          projected_exponential.shape().size() != 2 ||
            projected_exponential.shape()[0] < krylov_dim || projected_exponential.shape()[1] < 1,
          "[ERROR][Lanczos_Exp] projected exponential has incompatible shape.%s", "\n");

        UniTensor out = projected_exponential.at({0, 0}) * input_norm * basis[0];
        for (cytnx_uint64 i = 1; i < krylov_dim; ++i) {
          out += projected_exponential.at({i, 0}) * input_norm * basis[i];
        }
        out.set_rowrank_(basis[0].rowrank());
        return out;
      }

      double scalar_abs_internal(const Scalar &s) { return static_cast<double>(s.abs()); }

      Scalar real_scalar_for_dtype_internal(const double value, const unsigned int dtype) {
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          return Scalar(static_cast<cytnx_float>(value));
        }
        return Scalar(static_cast<cytnx_double>(value));
      }

      Tensor projected_exponential_internal(const Tensor &hp, const Scalar &tau) {
        if (Type.is_complex(tau.dtype())) {
          const cytnx_complex128 tau_cd(static_cast<cytnx_double>(tau.real()),
                                        static_cast<cytnx_double>(tau.imag()));
          return linalg::ExpH(hp, tau_cd, cytnx_complex128(0.0, 0.0));
        }
        return linalg::ExpH(hp, static_cast<cytnx_double>(tau.real()), cytnx_double(0.0));
      }

      double krylov_exponential_error_bound_internal(const double input_norm,
                                                     const double beta_next,
                                                     const long double gamma_product,
                                                     const double tau_abs,
                                                     const cytnx_uint32 krylov_dim) {
        // Jawecki/Auzinger/Koch 2020, Theorem 1: beta_{m+1,m} gamma_m |tau|^m / m!.
        // The paper assumes a unit starting vector, so scale back to the input norm here.
        if (beta_next == 0.0 || gamma_product == 0.0L || tau_abs == 0.0) {
          return 0.0;
        }
        const long double log_estimate =
          std::log(static_cast<long double>(input_norm)) +
          std::log(static_cast<long double>(beta_next)) + std::log(gamma_product) +
          static_cast<long double>(krylov_dim) * std::log(static_cast<long double>(tau_abs)) -
          std::lgammal(static_cast<long double>(krylov_dim) + 1.0L);
        if (log_estimate <= std::log(std::numeric_limits<double>::min())) {
          return 0.0;
        }
        if (log_estimate >= std::log(std::numeric_limits<double>::max())) {
          return std::numeric_limits<double>::infinity();
        }
        return static_cast<double>(std::exp(log_estimate));
      }

      Tensor resize_mat_internal(const Tensor &src, const cytnx_uint64 r, const cytnx_uint64 c) {
        const auto min_r = std::min(r, src.shape()[0]);
        const auto min_c = std::min(c, src.shape()[1]);
        // Tensor dst = src[{ac::range(0,min_r),ac::range(0,min_c)}];

        Tensor dst = Tensor({min_r, min_c}, src.dtype(), src.device(), false);
        char *tgt = (char *)dst.storage().data();
        char *csc = (char *)src.storage().data();
        unsigned long long Offset_csc = Type.typeSize(src.dtype()) * src.shape()[1];
        unsigned long long Offset_tgt = Type.typeSize(src.dtype()) * min_c;
        for (auto i = 0; i < min_r; ++i) {
          memcpy(tgt + Offset_tgt * i, csc + Offset_csc * i, Type.typeSize(src.dtype()) * min_c);
        }

        return dst;
      }

      // BiCGSTAB method to solve the linear equation
      // ref: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
      UniTensor invert_biCGSTAB_internal(LinOp *Hop, const UniTensor &b, const UniTensor &Tin,
                                         const int &k, const double &CvgCrit = 1.0e-12,
                                         const unsigned int &Maxiter = 10000) {
        // the operation (I + Hop/k) on A
        auto I_plus_A_Op = [&](UniTensor A) {
          return ((Hop->matvec(A)) / k + A).relabel_(b.labels());
        };
        // the residuals of (b - (I + Hop/k)x)
        auto r = (b - I_plus_A_Op(Tin)).relabel_(b.labels());
        // choose r0_hat = r
        auto r0 = r;
        auto x = Tin;
        // choose p = (r0_hat, r)
        auto p = Dot_internal(r0, r);
        // choose pv = r
        auto pv = r;

        // to reduce the variables used, replace p[i]->p, p[i-1]->p_old, etc.
        auto p_old = p;
        auto pv_old = pv;
        auto x_old = x;
        auto r_old = r;

        // all of logic here is same
        // as:https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
        for (int i = 1; i < Maxiter; ++i) {
          auto v = I_plus_A_Op(pv_old);
          auto a = p_old / Dot_internal(r0, v);
          auto h = (x_old + a * pv_old).relabel_(b.labels());
          auto s = (r_old - a * v).relabel_(b.labels());
          if (abs(Dot_internal(s, s)) < CvgCrit) {
            x = h;
            break;
          }
          auto t = I_plus_A_Op(s);
          auto w = Dot_internal(t, s) / Dot_internal(t, t);
          x = (h + w * s).relabel_(b.labels());
          r = (s - w * t).relabel_(b.labels());
          if (abs(Dot_internal(r, r)) < CvgCrit) {
            break;
          }
          auto p = Dot_internal(r0, r);
          auto beta = (p / p_old) * (a / w);
          pv = (r + beta * (pv_old - w * v)).relabel_(b.labels());

          // update
          pv_old = pv;
          p_old = p;
          x_old = x;
          r_old = r;
        }
        return x;
      }

      // ref:  https://doi.org/10.48550/arXiv.1111.1491
      void Lanczos_Exp_Ut_internal_positive(UniTensor &out, LinOp *Hop, const UniTensor &Tin,
                                            const double &CvgCrit, const unsigned int &Maxiter,
                                            const bool &verbose) {
        double delta = CvgCrit;
        int k = static_cast<int>(std::log(1.0 / delta));
        k = k < Maxiter ? k : Maxiter;
        auto Op_apprx_norm =
          static_cast<double>(Hop->matvec(Tin).get_block_().flatten().Norm().item().real());
        double eps1 = std::exp(-(k * std::log(k) + std::log(1.0 + Op_apprx_norm)));

        std::vector<UniTensor> vs;
        Tensor as = zeros({(cytnx_uint64)k + 1, (cytnx_uint64)k + 1}, Tin.dtype(), Tin.device());

        // Initialized v0 = v
        auto v = Tin;
        auto v0 = v;
        auto Vk_shape = v0.shape();
        Vk_shape.insert(Vk_shape.begin(), 1);
        auto Vk = v0.get_block().reshape(Vk_shape);
        std::vector<UniTensor> Vs;
        Vs.push_back(v);

        // For i = 0 to k
        for (int i = 0; i <= k; ++i) {
          // Call the procedure Invert_A (v[i], k, eps1). The procedure returns a vector w[i], such
          // that,
          // |(I + A / k )^(−1) v[i] − w[i]| ≤ eps1 |v[i]| .
          auto w = invert_biCGSTAB_internal(Hop, v, v, k, eps1);
          // auto resi = ((Hop->matvec(w))/k + w).relabel_(v.labels()) - v;

          // For j = 0 to i
          for (int j = 0; j <= i; ++j) {
            // Let a[j,i] = <v[j], w[i]>
            as.at({(cytnx_uint64)j, (cytnx_uint64)i}) = Dot_internal(Vs.at(j), w);
          }
          // Define wp[i] = w[i] - \sum_{j=0}^{j=i} {a[j,i]v[j]}
          auto wp = w;
          for (int j = 0; j <= i; j++) {
            wp -= as.at({(cytnx_uint64)j, (cytnx_uint64)i}) * Vs.at(j);
          }
          // Let a[i+1, i] = |wp[i]|, v[i+1]=wp[i] / a[i+1, i]
          auto b = std::sqrt(double(Dot_internal(wp, wp).real()));
          if (i < k) {
            as.at({(cytnx_uint64)i + 1, (cytnx_uint64)i}) = b;
            v = wp / b;
            Vk.append(v.get_block_());
            Vs.push_back(v);
          }
          // For j = i+2 to k
          //   Let a[j,i] = 0
        }

        // Let V_k be the n × (k + 1) matrix whose columns are v[0],...,v[k] respectively.
        UniTensor Vk_ut(Vk);
        Vk_ut.set_rowrank_(1);
        auto VkDag_ut = Vk_ut.Dagger();  // index order is inverted here!
        // Let T_k be the (k + 1) × (k + 1) matrix a[i,j] i,j is {0,...,k} and Tk_hat = 1 / 2
        // (Tk^Dagger  + Tk).
        auto asT = as.permute({1, 0}).Conj().contiguous();
        auto Tk_hat = 0.5 * (asT + as);
        // Compute B = exp k · (I − Tk_hat^(−1) ) exactly and output the vector V_k*B*V_k^Dagger v.
        auto I = eye(k + 1);
        auto B_mat = linalg::ExpH(k * (I - linalg::InvM(Tk_hat)));
        /*
         *    |||
         *  |-----|
         *  | out |        =
         *  |_____|
         *
         *
         *    |||
         *  |-----|
         *  | V_k |
         *  |_____|
         *     |    kl:(k+1) * (k + 1)
         *     |
         *  |-----|
         *  |  B  |
         *  |_____|
         *     |    kr:(k+1) * (k + 1)
         *     |
         *  |------------|
         *  | V_k^Dagger |
         *  |____________|
         *    |||
         *  |-----|
         *  |  v0 |
         *  |_____|
         *
         */
        auto B = UniTensor(B_mat, false, 1, {"cytnx_internal_label_kl", "cytnx_internal_label_kr"});
        auto label_kl = B.labels()[0];
        auto label_kr = B.labels()[1];
        auto Vk_labels = v0.labels();
        Vk_labels.insert(Vk_labels.begin(), label_kl);
        Vk_ut.relabel_(Vk_labels);
        auto VkDag_labels =
          std::vector<std::string>(v0.labels().rbegin(), v0.labels().rend());  // inverted order
        VkDag_labels.push_back(label_kr);
        VkDag_ut.relabel_(VkDag_labels);

        out = Contracts({v0, VkDag_ut, B}, "", true);
        out = Contract(out, Vk_ut);
        out.set_rowrank_(v0.rowrank());
      }

      void Lanczos_Exp_Ut_internal(UniTensor &out, LinOp *Hop, const UniTensor &T, Scalar tau,
                                   const unsigned int output_dtype, const double &CvgCrit,
                                   const unsigned int &Maxiter, const bool &verbose,
                                   KrylovStats *stats) {
        const unsigned int input_dtype = T.dtype();
        const double eps = dtype_epsilon_internal(input_dtype);
        std::vector<UniTensor> vs;
        cytnx_uint32 vec_len = 1;
        if (T.uten_type() == UTenType.Dense) {
          vec_len = tensor_element_count_internal(T.get_block_());
        } else if (T.uten_type() == UTenType.Block || T.uten_type() == UTenType.BlockFermionic) {
          vec_len = 0;
          for (const auto &block : T.get_blocks_()) vec_len += tensor_element_count_internal(block);
        }
        cytnx_uint32 imp_maxiter = std::min(Maxiter, vec_len);
        if (stats) {
          stats->maxiter_used = imp_maxiter;
        }
        Tensor Hp =
          zeros({imp_maxiter, imp_maxiter}, krylov_matrix_dtype_internal(input_dtype), T.device());

        Tensor B_mat;
        // prepare initial tensor and normalize
        auto v = T.clone();
        auto v_nrm = std::sqrt(double(Dot_internal(v, v).real()));
        v = v / real_scalar_for_dtype_internal(v_nrm, input_dtype);
        const double tau_abs = scalar_abs_internal(tau);
        long double gamma_product = 1.0L;

        // first iteration
        auto wp = (Hop->matvec(v)).relabel_(v.labels());
        check_matvec_output_internal(wp, v);
        if (stats) {
          stats->matvec_count++;
        }
        auto alpha = Dot_internal(wp, v);
        Hp.at({0, 0}) = alpha;
        auto w = (wp - alpha * v).relabel_(v.labels());
        double beta_prev = 0.0;
        double beta_breakdown_scale = std::max(scalar_abs_internal(alpha), 1.0);
        if (stats && imp_maxiter == 1) {
          const auto beta = std::sqrt(double(Dot_internal(w, w).real()));
          stats->converged = true;
          stats->reason = "full_krylov_dimension";
          stats->final_beta = beta;
          stats->breakdown_tol = kBetaBreakdownRoundoff * eps * beta_breakdown_scale;
        }

        std::vector<UniTensor> Vs;
        Vs.push_back(v);
        UniTensor v_old;
        Tensor Hp_sub;

        for (cytnx_uint32 i = 1; i < imp_maxiter; ++i) {
          if (verbose) {
            std::cout << "Lanczos iteration:" << i << std::endl;
          }
          auto beta = std::sqrt(double(Dot_internal(w, w).real()));
          v_old = v.clone();
          auto local_scale = scalar_abs_internal(alpha) + std::abs(beta) + std::abs(beta_prev);
          beta_breakdown_scale = std::max(beta_breakdown_scale, std::max(local_scale, 1.0));
          auto beta_breakdown =
            kBetaBreakdownRoundoff * eps * beta_breakdown_scale * std::sqrt(static_cast<double>(i));
          if (beta > beta_breakdown) {
            v = (w / real_scalar_for_dtype_internal(beta, input_dtype)).relabel_(v.labels());
          } else {  // beta too small -> the norm of new vector too small. This vector cannot span
                    // the new dimension
            if (verbose) {
              std::cout << "beta too small. Break at iteration " << i << std::endl;
            }
            if (stats) {
              stats->converged = true;
              stats->reason = "breakdown";
              stats->final_beta = beta;
              stats->breakdown_tol = beta_breakdown;
            }
            break;
          }
          Vs.push_back(v);
          gamma_product *= static_cast<long double>(std::abs(beta));
          Hp.at({(cytnx_uint64)i, (cytnx_uint64)i - 1}) =
            Hp.at({(cytnx_uint64)i - 1, (cytnx_uint64)i}) = beta;
          wp = (Hop->matvec(v)).relabel_(v.labels());
          check_matvec_output_internal(wp, v);
          if (stats) {
            stats->matvec_count++;
          }
          alpha = Dot_internal(wp, v);
          Hp.at({(cytnx_uint64)i, (cytnx_uint64)i}) = alpha;
          w = (wp - alpha * v - real_scalar_for_dtype_internal(beta, input_dtype) * v_old)
                .relabel_(v.labels());
          beta_prev = beta;
          const auto beta_next = std::sqrt(double(Dot_internal(w, w).real()));
          const auto local_scale_next =
            scalar_abs_internal(alpha) + std::abs(beta_next) + std::abs(beta);
          beta_breakdown_scale = std::max(beta_breakdown_scale, std::max(local_scale_next, 1.0));
          const auto beta_breakdown_next = kBetaBreakdownRoundoff * eps * beta_breakdown_scale *
                                           std::sqrt(static_cast<double>(i + 1));

          // Converge check
          Hp_sub = resize_mat_internal(Hp, i + 1, i + 1);
          B_mat = projected_exponential_internal(Hp_sub, tau);
          auto error = krylov_exponential_error_bound_internal(v_nrm, beta_next, gamma_product,
                                                               tau_abs, i + 1);
          if (stats) {
            stats->final_error = error;
            stats->final_beta = beta_next;
            stats->breakdown_tol = beta_breakdown_next;
          }
          if (beta_next <= beta_breakdown_next) {
            if (stats) {
              stats->converged = true;
              stats->reason = "breakdown";
            }
            break;
          }
          if (error < CvgCrit) {
            if (stats) {
              stats->converged = true;
              stats->reason = "projected_exponential";
            }
            break;
          }
          if (i == imp_maxiter - 1) {
            if (Maxiter < vec_len) {
              cytnx_warning_msg(
                true,
                "[WARNING][Lanczos_Exp] Did not converge after Maxiter [%u] iterations; try "
                "increasing maxiter.",
                imp_maxiter);
            }
            if (stats) {
              stats->converged = Maxiter >= vec_len;
              stats->reason = Maxiter < vec_len ? "maxiter" : "full_krylov_dimension";
            }
            break;
          }
        }
        if (B_mat.dtype() == Type.Void) {
          Hp_sub = resize_mat_internal(Hp, Vs.size(), Vs.size());
          B_mat = projected_exponential_internal(Hp_sub, tau);
        }
        if (stats) {
          stats->iterations = Vs.size();
          stats->krylov_dim = Vs.size();
        }

        out = reconstruct_from_krylov_basis_internal(Vs, B_mat, v_nrm);
        cast_dense_output_internal(
          out, projected_exponential_output_dtype_internal(output_dtype, out.dtype()));
      }
    }  // unnamed namespace

    // Lanczos_Exp
    UniTensor Lanczos_Exp(LinOp *Hop, const UniTensor &Tin, const Scalar &tau,
                          const double &CvgCrit, const unsigned int &Maxiter, const bool &verbose) {
      // check device:
      cytnx_error_msg(Tin.device() != Device.cpu,
                      "[ERROR][Lanczos_Exp] Lanczos_Exp still does not support cuda devices.%s",
                      "\n");
      cytnx_error_msg(!Type.is_float(Tin.dtype()),
                      "[ERROR][Lanczos_Exp] Lanczos_Exp can only accept input tensors with "
                      "floating types (complex/real)%s",
                      "\n");

      // check criteria and maxiter:
      cytnx_error_msg(CvgCrit <= 0, "[ERROR][Lanczos_Exp] converge criteria must >0%s", "\n");
      cytnx_error_msg(Maxiter < 2, "[ERROR][Lanczos_Exp] Maxiter must >1%s", "\n");

      // check Tin should be rank-1:

      // The Krylov basis dtype is determined by the state and operator. A complex timestep only
      // makes the projected exponential and final output complex; it should not force a real
      // operator to accept complex Krylov vectors.
      UniTensor v0;
      v0 = Tin;
      const auto output_dtype = promoted_output_dtype_internal(Tin.dtype(), tau.dtype());

      UniTensor out;

      double _cvgcrit = CvgCrit;

      const double cvgcrit_floor = cvgcrit_floor_internal(v0.dtype());
      if (_cvgcrit < cvgcrit_floor) {
        _cvgcrit = cvgcrit_floor;
        cytnx_warning_msg(
          should_warn_cvgcrit_floor_internal(v0.dtype()),
          "[WARNING][Lanczos_Exp] CvgCrit cannot be smaller than %.8e for dtype %s, and is "
          "automatically raised to this value.%s",
          cvgcrit_floor, Type.getname(v0.dtype()).c_str(), "\n");
      }

      // Lanczos_Exp_Ut_internal_positive(out, Hop, v0, _cvgcrit, Maxiter, verbose);
      KrylovStats stats;
      stats.algorithm = "Lanczos_Exp";
      stats.maxiter_requested = Maxiter;
      stats.cvgcrit_requested = CvgCrit;
      stats.cvgcrit_used = _cvgcrit;
      stats.input_dtype = Tin.dtype();
      stats.op_dtype = Type.Void;
      stats.working_dtype = v0.dtype();

      Lanczos_Exp_Ut_internal(out, Hop, v0, tau, output_dtype, _cvgcrit, Maxiter, verbose, &stats);
      set_last_krylov_stats(stats);

      return out;

    }  // Lanczos_Exp entry point

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
