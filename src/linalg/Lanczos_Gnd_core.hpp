#ifndef CYTNX_LINALG_LANCZOS_GND_CORE_HPP_
#define CYTNX_LINALG_LANCZOS_GND_CORE_HPP_

#include "linalg.hpp"
#include "DenseMatrix_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace cytnx {
  namespace linalg {
    namespace internal {

      struct SymmetricEigensystem {
        std::vector<double> values;
        DenseMatrix<double> vectors;
      };

      inline double dtype_epsilon(const unsigned int dtype) {
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          return std::numeric_limits<float>::epsilon();
        }
        return std::numeric_limits<double>::epsilon();
      }

      inline double beta_breakdown_tolerance(const double scale, const double eps,
                                             const cytnx_uint64 krylov_dim) {
        constexpr double roundoff_multiplier = 100.0;
        return roundoff_multiplier * eps * std::max(scale, 1.0) *
               std::sqrt(static_cast<double>(std::max<cytnx_uint64>(krylov_dim, 1)));
      }

      inline unsigned int capped_nonrestarted_maxiter(const unsigned int maxiter,
                                                      const std::uint64_t vec_len) {
        constexpr unsigned int max_nonrestarted_maxiter = 100;
        const unsigned int finite_space_cap =
          vec_len < maxiter ? static_cast<unsigned int>(vec_len) : maxiter;
        static bool warning_issued = false;
        if (finite_space_cap > max_nonrestarted_maxiter && !warning_issued) {
          warning_issued = true;
          cytnx_warning_msg(
            true,
            "[WARNING][Lanczos_Gnd] Non-restarted Lanczos would use %u Krylov steps after the "
            "finite-space cap, which is unusually large. Capping Maxiter to %u. Use the "
            "ARPACK-backed Lanczos(..., which=\"SA\") entry point for generic ground-state "
            "eigensolver use.%s",
            finite_space_cap, max_nonrestarted_maxiter, "\n");
        }
        return std::min(finite_space_cap, max_nonrestarted_maxiter);
      }

      inline SymmetricEigensystem symmetric_jacobi_eigensystem(DenseMatrix<double> matrix) {
        const cytnx_uint64 n = matrix.rows();
        cytnx_error_msg(matrix.rows() != matrix.cols(),
                        "[ERROR][Lanczos_Gnd] projected matrix must be square.%s", "\n");

        DenseMatrix<double> vectors(n, n);
        for (cytnx_uint64 i = 0; i < n; ++i) {
          vectors(i, i) = 1.0;
        }

        if (n > 1) {
          double scale = 1.0;
          for (cytnx_uint64 i = 0; i < n; ++i) {
            for (cytnx_uint64 j = 0; j < n; ++j) {
              scale = std::max(scale, std::abs(matrix(i, j)));
            }
          }
          const double tolerance = 100.0 * std::numeric_limits<double>::epsilon() * scale;
          const cytnx_uint64 max_sweeps = std::max<cytnx_uint64>(32 * n * n, 64);

          for (cytnx_uint64 sweep = 0; sweep < max_sweeps; ++sweep) {
            cytnx_uint64 p = 0;
            cytnx_uint64 q = 1;
            double largest = std::abs(matrix(p, q));
            for (cytnx_uint64 i = 0; i < n; ++i) {
              for (cytnx_uint64 j = i + 1; j < n; ++j) {
                const double candidate = std::abs(matrix(i, j));
                if (candidate > largest) {
                  largest = candidate;
                  p = i;
                  q = j;
                }
              }
            }
            if (largest <= tolerance) break;

            const double app = matrix(p, p);
            const double aqq = matrix(q, q);
            const double apq = matrix(p, q);
            const double tau = (aqq - app) / (2.0 * apq);
            const double sign_tau = tau >= 0.0 ? 1.0 : -1.0;
            const double t = sign_tau / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
            const double c = 1.0 / std::sqrt(1.0 + t * t);
            const double s = t * c;

            for (cytnx_uint64 k = 0; k < n; ++k) {
              if (k == p || k == q) continue;
              const double akp = matrix(k, p);
              const double akq = matrix(k, q);
              matrix(k, p) = c * akp - s * akq;
              matrix(p, k) = matrix(k, p);
              matrix(k, q) = s * akp + c * akq;
              matrix(q, k) = matrix(k, q);
            }

            matrix(p, p) = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            matrix(q, q) = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            matrix(p, q) = 0.0;
            matrix(q, p) = 0.0;

            for (cytnx_uint64 k = 0; k < n; ++k) {
              const double vkp = vectors(k, p);
              const double vkq = vectors(k, q);
              vectors(k, p) = c * vkp - s * vkq;
              vectors(k, q) = s * vkp + c * vkq;
            }
          }
        }

        std::vector<cytnx_uint64> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](const cytnx_uint64 lhs, const cytnx_uint64 rhs) {
          return matrix(lhs, lhs) < matrix(rhs, rhs);
        });

        SymmetricEigensystem result;
        result.values.resize(n);
        result.vectors = DenseMatrix<double>(n, n);
        for (cytnx_uint64 col = 0; col < n; ++col) {
          const cytnx_uint64 source_col = order[col];
          result.values[col] = matrix(source_col, source_col);
          for (cytnx_uint64 row = 0; row < n; ++row) {
            result.vectors(row, col) = vectors(row, source_col);
          }
        }
        return result;
      }

      inline SymmetricEigensystem tridiagonal_eigensystem(const std::vector<double> &alpha,
                                                          const std::vector<double> &beta) {
        const cytnx_uint64 n = alpha.size();
        DenseMatrix<double> projected(n, n);
        for (cytnx_uint64 i = 0; i < n; ++i) {
          projected(i, i) = alpha[i];
          if (i + 1 < n) {
            projected(i, i + 1) = beta[i];
            projected(i + 1, i) = beta[i];
          }
        }
        return symmetric_jacobi_eigensystem(projected);
      }

      template <typename Vector>
      struct LanczosGroundResult {
        double eigenvalue = 0.0;
        Vector eigenvector;
        bool has_eigenvector = false;
      };

      template <typename Ops>
      LanczosGroundResult<typename Ops::vector_type> lanczos_ground_state(
        Ops &ops, const typename Ops::vector_type &initial, const bool compute_vector,
        const double residual_tol, const unsigned int maxiter, const bool verbose,
        KrylovStats *stats) {
        using Vector = typename Ops::vector_type;

        cytnx_error_msg(residual_tol <= 0.0,
                        "[ERROR][Lanczos_Gnd] residual_tol must be positive.%s", "\n");
        cytnx_error_msg(maxiter < 1, "[ERROR][Lanczos_Gnd] Maxiter must be at least 1.%s", "\n");

        const std::uint64_t vec_len = ops.dimension();
        cytnx_error_msg(vec_len == 0, "[ERROR][Lanczos_Gnd] LinOp::nx() must be positive.%s", "\n");
        const unsigned int imp_maxiter = capped_nonrestarted_maxiter(maxiter, vec_len);
        const cytnx_uint64 min_krylov_dim =
          std::min<cytnx_uint64>({4, static_cast<cytnx_uint64>(imp_maxiter), vec_len});
        const double eps = dtype_epsilon(ops.working_dtype());
        if (stats) {
          stats->maxiter_used = imp_maxiter;
          stats->residual_tol_requested = residual_tol;
          stats->residual_tol_used = residual_tol;
        }

        const double initial_norm = ops.norm(initial);
        cytnx_error_msg(initial_norm == 0.0, "[ERROR][Lanczos_Gnd] initial vector has zero norm.%s",
                        "\n");

        Vector q = ops.scale(initial, 1.0 / initial_norm);
        Vector q_prev;
        std::vector<Vector> basis;
        if (compute_vector) {
          basis.push_back(q);
        }

        std::vector<double> alpha;
        std::vector<double> beta_previous;
        alpha.reserve(imp_maxiter);
        beta_previous.reserve(imp_maxiter > 0 ? imp_maxiter - 1 : 0);

        double beta = 0.0;
        double beta_scale = 1.0;
        SymmetricEigensystem projected;
        cytnx_uint64 ground_index = 0;
        std::string stop_reason = "maxiter";
        bool converged = false;

        for (unsigned int j = 0; j < imp_maxiter; ++j) {
          Vector residual = ops.matvec(q);
          ops.check_matvec_output(residual, q);
          if (stats) {
            stats->matvec_count++;
          }

          const double alpha_j = ops.hermitian_inner_product_real(residual, q);
          alpha.push_back(alpha_j);
          ops.axpy(residual, -alpha_j, q);
          if (j > 0) {
            ops.axpy(residual, -beta_previous[j - 1], q_prev);
          }

          beta = ops.norm(residual);
          beta_scale = std::max(beta_scale, std::abs(alpha_j));
          if (j > 0) {
            beta_scale = std::max(beta_scale, std::abs(beta_previous[j - 1]));
          }
          const double breakdown_tol = beta_breakdown_tolerance(beta_scale, eps, alpha.size());
          projected = tridiagonal_eigensystem(alpha, beta_previous);
          ground_index = 0;
          const double theta = projected.values[ground_index];
          const double residual_bound = beta * std::abs(projected.vectors(alpha.size() - 1, 0));
          const double residual_target = residual_tol * std::max(1.0, std::abs(theta));
          if (stats) {
            stats->iterations = alpha.size();
            stats->krylov_dim = alpha.size();
            stats->final_error = residual_bound;
            stats->final_residual = residual_bound;
            stats->final_beta = beta;
            stats->breakdown_tol = breakdown_tol;
          }

          if (verbose) {
            std::cout << "iter[" << j << "] Enr: " << theta
                      << ", residual bound: " << residual_bound << std::endl;
          }

          const bool happy_breakdown = beta <= breakdown_tol;
          const bool full_krylov_dimension = alpha.size() >= vec_len;
          const bool residual_converged =
            alpha.size() >= min_krylov_dim && residual_bound <= residual_target;
          if (happy_breakdown || full_krylov_dimension || residual_converged) {
            converged = true;
            if (happy_breakdown) {
              stop_reason = "breakdown";
            } else if (full_krylov_dimension) {
              stop_reason = "full_krylov_dimension";
            } else {
              stop_reason = "residual";
            }
            break;
          }

          if (j + 1 == imp_maxiter) {
            break;
          }

          q_prev = q;
          q = ops.scale(residual, 1.0 / beta);
          beta_previous.push_back(beta);
          if (compute_vector) {
            basis.push_back(q);
          }
        }

        if (stats) {
          stats->converged = converged;
          stats->reason = stop_reason;
        }

        LanczosGroundResult<Vector> result;
        result.eigenvalue = projected.values[ground_index];
        if (compute_vector) {
          result.eigenvector = ops.scale(basis[0], projected.vectors(0, ground_index));
          for (cytnx_uint64 i = 1; i < basis.size(); ++i) {
            ops.axpy(result.eigenvector, projected.vectors(i, ground_index), basis[i]);
          }
          result.has_eigenvector = true;
        }
        return result;
      }

    }  // namespace internal
  }  // namespace linalg
}  // namespace cytnx

#endif  // CYTNX_LINALG_LANCZOS_GND_CORE_HPP_
