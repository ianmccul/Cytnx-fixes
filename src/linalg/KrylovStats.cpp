#include "linalg.hpp"

namespace cytnx {
  namespace linalg {

    namespace {
      thread_local KrylovStats last_stats;
      thread_local KrylovStats total_stats;

      void add_to_total(const KrylovStats &stats) {
        total_stats.algorithm = "all";
        total_stats.reason = stats.reason;
        total_stats.converged = stats.converged;
        total_stats.matvec_count += stats.matvec_count;
        total_stats.iterations += stats.iterations;
        total_stats.krylov_dim += stats.krylov_dim;
        total_stats.maxiter_requested += stats.maxiter_requested;
        total_stats.maxiter_used += stats.maxiter_used;
        total_stats.final_error = stats.final_error;
        total_stats.final_beta = stats.final_beta;
        total_stats.breakdown_tol = stats.breakdown_tol;
        total_stats.input_dtype = stats.input_dtype;
        total_stats.working_dtype = stats.working_dtype;
      }
    }  // namespace

    KrylovStats last_krylov_stats() { return last_stats; }

    KrylovStats krylov_stats() { return total_stats; }

    void set_last_krylov_stats(const KrylovStats &stats) {
      last_stats = stats;
      add_to_total(stats);
    }

    void clear_krylov_stats() {
      last_stats = KrylovStats();
      total_stats = KrylovStats();
    }

  }  // namespace linalg
}  // namespace cytnx
