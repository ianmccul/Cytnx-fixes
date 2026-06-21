#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "UniTensor.hpp"
#include "utils/vec_print.hpp"
#include "backend/arpack_wrapper.hpp"
#include <iomanip>

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;

    // T_ten: Tensor or UniTensor
    template <typename T, typename T_ten>
    static T *get_obj_data_ptr(const T_ten &buffer, const cytnx_int32 bk_idx = 0) {
      if constexpr (std::is_same_v<T_ten, UniTensor>) {
        if (buffer.uten_type() == UTenType.Block || buffer.uten_type() == UTenType.BlockFermionic) {
          return buffer.get_blocks_()[bk_idx].template ptr_as<T>();
        } else if (buffer.uten_type() == UTenType.Dense) {
          return buffer.get_block_().template ptr_as<T>();
        }
      } else if constexpr (std::is_same_v<T_ten, Tensor>) {
        return buffer.template ptr_as<T>();
      }
    }

    static cytnx_int64 get_dim(const Tensor &T) {
      auto shape = T.shape();
      cytnx_int64 dim = 1;
      for (auto &x : shape) {
        dim *= x;
      }
      return dim;
    }

    static cytnx_int64 get_elem_num(const UniTensor &UT) {
      if (UT.uten_type() == UTenType.Block || UT.uten_type() == UTenType.BlockFermionic) {
        cytnx_int64 dim = 0;
        auto &blocks = UT.get_blocks_();
        for (int i = 0; i < blocks.size(); ++i) {
          dim += get_dim(blocks[i]);
        }
        return dim;
      } else if (UT.uten_type() == UTenType.Dense) {
        return get_dim(UT.get_block_());
      }
    }

    static unsigned int promoted_working_dtype(const unsigned int input_dtype,
                                               const unsigned int op_dtype,
                                               const unsigned int default_dtype = Type.Double) {
      if (input_dtype == Type.Void) {
        return op_dtype == Type.Void ? default_dtype : op_dtype;
      }
      if (op_dtype == Type.Void) {
        return input_dtype;
      }
      return Type.type_promote(input_dtype, op_dtype);
    }

    template <typename T, typename T_ten>
    static void pass_data_UT(T_ten &UT, T *data_ptr, bool to_UT) {
      auto device = UT.device();
      if constexpr (std::is_same_v<T_ten, UniTensor>) {
        if (UT.uten_type() == UTenType.Block || UT.uten_type() == UTenType.BlockFermionic) {
          auto &blocks = UT.get_blocks_();
          for (auto &block : blocks) {
            auto dim = get_dim(block);
            T *UT_data = get_obj_data_ptr<T, Tensor>(block);
            if (to_UT) {
              if (device == Device.cpu) {
                memcpy(UT_data, data_ptr, dim * sizeof(T));
              } else {
  #ifdef UNI_GPU
                checkCudaErrors(
                  cudaMemcpy(UT_data, data_ptr, dim * sizeof(T), cudaMemcpyHostToDevice));
  #endif
              }
            } else {
              if (device == Device.cpu) {
                memcpy(data_ptr, UT_data, dim * sizeof(T));
              } else {
  #ifdef UNI_GPU
                checkCudaErrors(
                  cudaMemcpy(data_ptr, UT_data, dim * sizeof(T), cudaMemcpyDeviceToHost));
  #endif
              }
            }
            data_ptr += dim;
          }
          return;
        } else if (UT.uten_type() == UTenType.Dense) {
          auto &block = UT.get_block_();
          auto dim = get_dim(block);
          T *UT_data = get_obj_data_ptr<T, Tensor>(block);
          if (to_UT) {
            if (device == Device.cpu) {
              memcpy(UT_data, data_ptr, dim * sizeof(T));
            } else {
  #ifdef UNI_GPU
              checkCudaErrors(
                cudaMemcpy(UT_data, data_ptr, dim * sizeof(T), cudaMemcpyHostToDevice));
  #endif
            }
          } else {
            if (device == Device.cpu) {
              memcpy(data_ptr, UT_data, dim * sizeof(T));
            } else {
  #ifdef UNI_GPU
              checkCudaErrors(
                cudaMemcpy(data_ptr, UT_data, dim * sizeof(T), cudaMemcpyDeviceToHost));
  #endif
            }
          }
          return;
        }
      } else if constexpr (std::is_same_v<T_ten, Tensor>) {
        auto dim = get_dim(UT);
        T *UT_data = get_obj_data_ptr<T, Tensor>(UT);
        if (to_UT) {
          if (device == Device.cpu) {
            memcpy(UT_data, data_ptr, dim * sizeof(T));
          } else {
  #ifdef UNI_GPU
            checkCudaErrors(cudaMemcpy(UT_data, data_ptr, dim * sizeof(T), cudaMemcpyHostToDevice));
  #endif
          }
        } else {
          if (device == Device.cpu) {
            memcpy(data_ptr, UT_data, dim * sizeof(T));
          } else {
  #ifdef UNI_GPU
            checkCudaErrors(cudaMemcpy(data_ptr, UT_data, dim * sizeof(T), cudaMemcpyDeviceToHost));
  #endif
          }
        }
        return;
      }
    }

    std::vector<Tensor> Lanczos(LinOp *Hop, const Tensor &Tin, const std::string method,
                                double residual_tol, unsigned int Maxiter, cytnx_uint64 k,
                                bool is_V, bool is_row, cytnx_uint32 max_krydim, bool verbose) {
      if (method == "ER") {
        return Lanczos_ER(Hop, k, is_V, Maxiter, residual_tol, is_row, Tin, max_krydim, verbose);
      } else if (method == "Gnd") {
        cytnx_error_msg(k > 1, "[ERROR][Lanczos] Only k = 1 is supported for 'Gnd' method.%s",
                        "\n");
        cytnx_warning_msg(
          max_krydim > 0,
          "[WARNING][Lanczos] max_krydim > 0 while it is irrelevant when using 'Gnd' method.%s",
          "\n");
        return Lanczos_Gnd(Hop, residual_tol, is_V, Tin, verbose, Maxiter);
      } else {
        cytnx_error_msg(
          true,
          "[ERROR][Lanczos] Invalid Lanczos method. The legacy method='ER' implementation has "
          "been disabled. For general Hermitian eigenvalue problems, use the ARPACK-backed "
          "Lanczos(..., which=\"SA\") entry point for the smallest algebraic eigenvalue, or choose "
          "another ARPACK 'which' selector as needed. The method='Gnd' path is a non-restarted "
          "Lanczos routine intended for specialized local ground-state solves where bounded "
          "matvec count is more important than standalone eigensolver semantics.%s",
          "\n");
        return std::vector<Tensor>();
      }
    }  // Lanczos

    std::vector<UniTensor> Lanczos(LinOp *Hop, const UniTensor &Tin, const std::string method,
                                   double residual_tol, unsigned int Maxiter, cytnx_uint64 k,
                                   bool is_V, bool is_row, cytnx_uint32 max_krydim, bool verbose) {
      if (method == "ER") {
        KrylovStats stats;
        stats.algorithm = "Lanczos_ER_Ut";
        stats.converged = false;
        stats.reason = "disabled";
        stats.maxiter_requested = Maxiter;
        stats.maxiter_used = Maxiter;
        stats.cvgcrit_requested = residual_tol;
        stats.cvgcrit_used = residual_tol;
        stats.residual_tol_requested = residual_tol;
        stats.residual_tol_used = residual_tol;
        stats.krylov_dim = max_krydim;
        stats.input_dtype = Tin.dtype();
        stats.working_dtype = Tin.dtype();
        set_last_krylov_stats(stats);
        throw std::runtime_error(
          "[ERROR][Lanczos] Lanczos method 'ER' has been disabled because the old implementation "
          "is numerically incorrect. For general Hermitian eigenvalue problems, use the "
          "ARPACK-backed Lanczos(..., which=\"SA\") entry point for the smallest algebraic "
          "eigenvalue, or choose another ARPACK 'which' selector as needed. The method='Gnd' path "
          "is a non-restarted Lanczos routine intended for specialized local ground-state solves "
          "where bounded matvec count is more important than standalone eigensolver semantics.");
      } else if (method == "Gnd") {
        cytnx_error_msg(k > 1, "[ERROR][Lanczos] Only k = 1 is supported for 'Gnd' method.%s",
                        "\n");
        cytnx_warning_msg(
          max_krydim > 0,
          "[WARNING][Lanczos] max_krydim > 0 while it is irrelevant when using 'Gnd' method.%s",
          "\n");
        return Lanczos_Gnd_Ut(Hop, Tin, residual_tol, is_V, verbose, Maxiter);
      } else {
        cytnx_error_msg(
          true,
          "[ERROR][Lanczos] Invalid Lanczos method. The legacy method='ER' implementation has "
          "been disabled. For general Hermitian eigenvalue problems, use the ARPACK-backed "
          "Lanczos(..., which=\"SA\") entry point for the smallest algebraic eigenvalue, or choose "
          "another ARPACK 'which' selector as needed. The method='Gnd' path is a non-restarted "
          "Lanczos routine intended for specialized local ground-state solves where bounded "
          "matvec count is more important than standalone eigensolver semantics.%s",
          "\n");
        return std::vector<UniTensor>();
      }
    }  // Lanczos

    template <typename T>
    static void clean_arpack_buffer(T *&resid, T *&v, T *&workd, T *&workl, cytnx_int32 *&select,
                                    T *&d, T *&z) {
      delete[] v;
      v = nullptr;
      delete[] resid;
      resid = nullptr;
      delete[] workd;
      workd = nullptr;
      delete[] workl;
      workl = nullptr;
      delete[] select;
      select = nullptr;
      delete[] d;
      d = nullptr;
      delete[] z;
      z = nullptr;
    }

    // T_ten: Tensor or UniTensor
    template <typename T, typename T_ten>
    static void matvec(LinOp *Hop, T_ten &buffer, T *v_in, T *v_out, bool &dtype_warning_issued) {
      buffer.contiguous_();
      pass_data_UT<T, T_ten>(buffer, v_in, true);
      auto nextTens = Hop->matvec(buffer);
      if (nextTens.dtype() != buffer.dtype()) {
        cytnx_error_msg(
          Type.type_promote(nextTens.dtype(), buffer.dtype()) != buffer.dtype(),
          "[ERROR][Lanczos], matvec returned dtype %s, which cannot be safely represented in the "
          "ARPACK workspace dtype %s. Use a wider input vector or LinOp dtype hint.%s",
          Type.getname(nextTens.dtype()).c_str(), Type.getname(buffer.dtype()).c_str(), "\n");
        if (!dtype_warning_issued) {
          cytnx_warning_msg(
            true,
            "[WARNING][Lanczos], matvec returned dtype %s while the ARPACK workspace dtype is %s. "
            "Casting the output; the LinOp dtype hint may be wider than the actual operator "
            "output.%s",
            Type.getname(nextTens.dtype()).c_str(), Type.getname(buffer.dtype()).c_str(), "\n");
          dtype_warning_issued = true;
        }
      }
      nextTens = nextTens.astype(buffer.dtype());
      // Resolve any pending fermionic signflip so the raw block data copied out below corresponds
      // to all signflips applied; no-op for bosonic/dense tensors.
      if constexpr (std::is_same_v<T_ten, UniTensor>) nextTens.apply_();
      nextTens.contiguous_();
      pass_data_UT<T, T_ten>(nextTens, v_out, false);
    }

    // sort eignvalues, it seems that arpack will not sort it...
    template <typename T>
    std::vector<cytnx_int32> sort_indices(const T *d, const char *which, const cytnx_int32 k) {
      char large_or_small = which[0];  //'S' or 'L'
      char metric_type = which[1];  //'M', or 'A'

      // get the metric distance
      auto metric = std::vector<T>(k, 0);
      if (metric_type == 'M') {
        for (cytnx_int32 i = 0; i < k; ++i) metric[i] = std::abs(d[i]);
      } else if (metric_type == 'A') {
        for (cytnx_int32 i = 0; i < k; ++i) metric[i] = d[i];
      } else {
        cytnx_error_msg(true, "[ERROR][Lanczos], Not supprted 'which'.%s", "\n");
      }  // never

      std::vector<cytnx_int32> indices(k);
      std::iota(indices.begin(), indices.end(), 0);  // fills with 0, 1, ..., k-1

      // smallest or largest
      bool is_small = (large_or_small == 'S');
      std::sort(indices.begin(), indices.end(), [&](cytnx_int32 i, cytnx_int32 j) {
        if (is_small) {
          return metric[i] < metric[j];
        } else {
          return metric[i] > metric[j];
        }
      });
      return indices;
    }

    // T_ten: Tensor or UniTensor
    // T: cytnx_double or cytnx_float
    template <typename T, typename T_ten>
    void _Lanczos_internal(std::vector<T_ten> &out, LinOp *Hop, const T_ten &UT_init,
                           const std::string which_str, const cytnx_uint64 &maxiter,
                           const cytnx_double &CvgCrit, const cytnx_uint64 &k,
                           const cytnx_bool &is_V, const cytnx_int32 &ncv_in,
                           const cytnx_bool &verbose, KrylovStats *stats) {
      std::function<void(cytnx_int32 * ido, char *bmat, cytnx_int32 *n, char *which,
                         cytnx_int32 *nev, T *tol, T *resid, cytnx_int32 *ncv, T *v,
                         cytnx_int32 *ldv, cytnx_int32 *iparam, cytnx_int32 *ipntr, T *workd,
                         T *workl, cytnx_int32 *lworkl, cytnx_int32 *info)>
        func_xsaupd;
      std::function<void(cytnx_int32 * rvec, char *howmny, cytnx_int32 *select, T *d, T *z,
                         cytnx_int32 *ldz, T *sigma, char *bmat, cytnx_int32 *n, char *which,
                         cytnx_int32 *nev, T *tol, T *resid, cytnx_int32 *ncv, T *v,
                         cytnx_int32 *ldv, cytnx_int32 *iparam, cytnx_int32 *ipntr, T *workd,
                         T *workl, cytnx_int32 *lworkl, cytnx_int32 *info)>
        func_xseupd;
      if constexpr (std::is_same_v<T, cytnx_double>) {
        func_xsaupd = arpack::dsaupd_;
        func_xseupd = arpack::dseupd_;
      } else if constexpr (std::is_same_v<T, cytnx_float>) {
        func_xsaupd = arpack::ssaupd_;
        func_xseupd = arpack::sseupd_;
      } else {
        static_assert(
          std::is_same_v<T, cytnx_double> || std::is_same_v<T, cytnx_float>,
          "Unsupported template types for _Lanczos_internal. T must be double/float type.");
      }
      cytnx_int32 dim = Hop->nx();
      cytnx_int32 nev = k;
      cytnx_int32 ido = 0;  /// reverse communication parameter, must be zero before iteration
      char bmat = 'I';  ///'I': standard eigenproblem, 'G': generalized eigenproblem
      char which[2];
      std::memcpy(which, which_str.data(), 2);  /// type of asked eigenvalues
      cytnx_int32 info = 0;
      T tol = static_cast<T>(CvgCrit);
      cytnx_int32 iparam[11];
      iparam[0] = 1;  /// Specifies the shift strategy (1->exact)
      iparam[2] = maxiter;  /// Maximum number of iterations
      iparam[3] = 1;  // current code only for 1
      iparam[6] = 1;  /// Sets the mode of d(s)saupd.
      cytnx_int32 ipntr[14];
      T *resid = new T[dim];
      T_ten buffer_UT = UT_init.clone();
      // Pin the Krylov vectors to the sign frame with all signflips applied so the raw block data
      // the iteration operates on is consistent; no-op for bosonic/dense tensors.
      if constexpr (std::is_same_v<T_ten, UniTensor>) buffer_UT.apply_();
      cytnx_bool ifinit = true;  // not allow for false, currently
      if (ifinit) {
        info = 1;
        pass_data_UT<T, T_ten>(buffer_UT, resid, false);
      } else {
        info = 0;
      }
      // If info = 0, a randomly initial residual vector is used.
      // If info = 1, resid contains the initial guess vector provided by user or from previous run
      // on output, resid contains the final residual vector

      // cytnx_int32 ncv = cytnx_int32(sqrt(dim));
      // ncv = (ncv > dim) ? dim : ncv; ///the number of Ritz vector, nev+2 <= ncv <= dim
      cytnx_int32 ncv = ncv_in == 0 ? std::min(dim, 2 * nev + 10) : ncv_in;
      cytnx_int32 ldv = dim;  /// leading dimension of v
      T *v = new T[dim * ncv];
      T *workd = new T[3 * dim];
      cytnx_int32 lworkl = ncv * (ncv + 8);  /// LWORKL must be at least NCV**2 + 8*NCV .
      T *workl = new T[lworkl];
      cytnx_int32 *select = new cytnx_int32[ncv];
      T *d = new T[nev];
      T *z = new T[dim * nev];

      T sigma;

      /// start iteration
      bool dtype_warning_issued = false;
      while (true) {
        func_xsaupd(&ido, &bmat, &dim, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr,
                    workd, workl, &lworkl, &info);
        if (ido == -1 || ido == 1) {
          matvec(Hop, buffer_UT, &workd[ipntr[0] - 1], &workd[ipntr[1] - 1], dtype_warning_issued);
          if (stats) {
            stats->matvec_count++;
          }
        } else if (ido == 99) {
          break;
        } else {
          clean_arpack_buffer(resid, v, workd, workl, select, d, z);
          cytnx_error_msg(true, "[ERROR][Lanczos], Error: d(s)saupd_ IDO = %d\n", ido);
        }
      }
      if (info < 0) {
        clean_arpack_buffer(resid, v, workd, workl, select, d, z);
        cytnx_error_msg(true, "[ERROR][Lanczos], Error: d(s)saupd_ INFO = %d\n", info);
      } else if (info == 1) {
        if (verbose) {
          cytnx_warning_msg(true,
                            "[WARNING][Anoldi]: Maximum number of iterations reached."
                            "Try to increase the maximum number of iteratoin.\n"
                            "d(s)saupd info = %d\n",
                            info);
        }
      } else if (info == 3) {
        clean_arpack_buffer(resid, v, workd, workl, select, d, z);
        cytnx_error_msg(true,
                        "[ERROR][Lanczos], No shifts could be applied during a cycle of "
                        "the Implicitly Restarted Lanczos process\n ."
                        "d(s)saupd info = %d\n",
                        info);
      }
      if (stats) {
        stats->converged = info == 0;
        stats->reason = info == 0 ? "converged" : "maxiter";
        stats->iterations = iparam[2];
        stats->krylov_dim = ncv;
        stats->maxiter_used = maxiter;
      }
      /// calculate eigenvalue and eigenvector by dseupd
      cytnx_int32 rvec = static_cast<cytnx_int32>(is_V);  /// 0: only eigenvalue, 1: also
                                                          /// eigenvector
      char howmny =
        'A';  /// how many eigenvectors to calculate: 'A' => nev eigenvectors
              ///  when howmny == 'A', this is used as workspace to reorder the eigenvectors
      func_xseupd(&rvec, &howmny, select, d, z, &ldv, &sigma, &bmat, &dim, which, &nev, &tol, resid,
                  &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
      if (info != 0) {
        clean_arpack_buffer(resid, v, workd, workl, select, d, z);
        cytnx_error_msg(true, "[ERROR][Lanczos], Error: d(s)seupd_ INFO = %d\n", info);
      }

      auto sorted_idx = sort_indices<T>(d, which, k);
      T *eigvals_data = get_obj_data_ptr<T, T_ten>(out[0]);
      for (cytnx_int32 ik = 0; ik < k; ++ik) {
        eigvals_data[ik] = d[sorted_idx[ik]];
      }

      if (is_V) {
        if constexpr (std::is_same_v<T_ten, UniTensor>) {
          for (cytnx_int32 ik = 0; ik < k; ++ik) {
            T *z_k_ptr = z + sorted_idx[ik] * dim;
            pass_data_UT<T, T_ten>(out[ik + 1], z_k_ptr, true);
          }
        } else if constexpr (std::is_same_v<T_ten, Tensor>) {
          T *tens_data = get_obj_data_ptr<T, T_ten>(out[1]);
          for (cytnx_int32 ik = 0; ik < k; ++ik) {
            T *tmp_data = tens_data + ik * dim;
            T *z_k_ptr = z + sorted_idx[ik] * dim;
            if (Hop->device() == Device.cpu) {
              memcpy(tmp_data, z_k_ptr, dim * sizeof(T));
            } else {
  #ifdef UNI_GPU
              checkCudaErrors(
                cudaMemcpy(tmp_data, z_k_ptr, dim * sizeof(T), cudaMemcpyHostToDevice));
  #endif
            }
          }
        }
      }

      clean_arpack_buffer(resid, v, workd, workl, select, d, z);
    }

    void _Lanczos(std::vector<UniTensor> &out, LinOp *Hop, const UniTensor &UT_init,
                  const std::string which, const cytnx_uint64 &maxiter, const double &CvgCrit,
                  const cytnx_uint64 &k, const bool &is_V, const cytnx_int32 &ncv,
                  const bool &verbose, KrylovStats *stats) {
      auto dtype = UT_init.dtype();
      auto device = Hop->device();
      auto eigvals_tens = zeros({k}, dtype, device);
      out.push_back(UniTensor(eigvals_tens));
      if (is_V) {
        // Output eigenvector templates: pass_data_UT fills these with the applied
        // (all-signflips-applied) eigenvector data, so any pending signflips must be cleared first;
        // apply_ is a no-op for bosonic/dense tensors.
        UniTensor UT_out = UT_init.clone();
        UT_out.apply_();
        for (cytnx_int32 ik = 0; ik < k; ++ik) out.push_back(UT_out.clone());
      }

      switch (dtype) {
        case Type.Double:
          _Lanczos_internal<cytnx_double, UniTensor>(out, Hop, UT_init, which, maxiter, CvgCrit, k,
                                                     is_V, ncv, verbose, stats);
          break;
        case Type.Float:
          _Lanczos_internal<cytnx_float, UniTensor>(out, Hop, UT_init, which, maxiter, CvgCrit, k,
                                                    is_V, ncv, verbose, stats);
          break;
      }
    }

    void _Lanczos(std::vector<Tensor> &out, LinOp *Hop, const Tensor &UT_init,
                  const std::string which, const cytnx_uint64 &maxiter, const double &CvgCrit,
                  const cytnx_uint64 &k, const bool &is_V, const cytnx_int32 &ncv,
                  const bool &verbose, KrylovStats *stats) {
      auto dtype = UT_init.dtype();
      auto device = Hop->device();
      auto eigvals_tens = zeros({k}, dtype, device);
      auto dim = Hop->nx();
      out[0] = eigvals_tens;
      if (is_V) {
        auto eigTens = k == 1 ? zeros({dim}, dtype, device) : zeros({k, dim}, dtype, device);
        out[1] = eigTens;
      }

      switch (dtype) {
        case Type.Double:
          _Lanczos_internal<cytnx_double, Tensor>(out, Hop, UT_init, which, maxiter, CvgCrit, k,
                                                  is_V, ncv, verbose, stats);
          break;
        case Type.Float:
          _Lanczos_internal<cytnx_float, Tensor>(out, Hop, UT_init, which, maxiter, CvgCrit, k,
                                                 is_V, ncv, verbose, stats);
          break;
      }
    }

    std::vector<Tensor> Lanczos(LinOp *Hop, const Tensor &T_init, const std::string which,
                                const cytnx_uint64 &maxiter, const double &cvg_crit,
                                const cytnx_uint64 &k, const bool &is_V, const cytnx_int32 &ncv,
                                const bool &verbose) {
      // check which
      std::vector<std::string> accept_which = {"LM", "LA", "SA"};
      if (std::find(accept_which.begin(), accept_which.end(), which) == accept_which.end()) {
        cytnx_error_msg(true, "[ERROR][Lanczos] 'which' should be 'LM', 'LA', 'SA'", "\n");
      }
      /// check k
      cytnx_error_msg(k < 1, "[ERROR][Lanczos] k should be >0%s", "\n");
      cytnx_error_msg(k > Hop->nx(),
                      "[ERROR][Lanczos] k can only be up to total dimension of input vector D%s",
                      "\n");

      // check Tin should be rank-1:
      auto _T_init = T_init.clone();
      if (T_init.dtype() == Type.Void) {
        _T_init = cytnx::random::normal({Hop->nx()}, 0, 1, Hop->device())
                    .astype(promoted_working_dtype(Type.Void, Hop->dtype()));
      } else {
        cytnx_error_msg(_T_init.shape().size() != 1, "[ERROR][Lanczos] Tin should be rank-1%s",
                        "\n");
        cytnx_error_msg(_T_init.shape()[0] != Hop->nx(),
                        "[ERROR][Lanczos] Tin should have dimension consistent with Hop: [%d] %s",
                        Hop->nx(), "\n");
        cytnx_error_msg(!Type.is_float(_T_init.dtype()),
                        "[ERROR][Lanczos] Tin should have floating dtype.%s", "\n");
        _T_init = _T_init.astype(promoted_working_dtype(_T_init.dtype(), Hop->dtype()));
      }

      if (Type.is_complex(_T_init.dtype())) {
        std::string tmp_which = which;
        if (which == "LA") {
          tmp_which = "LR";
        } else if (which == "SA") {
          tmp_which = "SR";
        }
        return Arnoldi(Hop, _T_init, tmp_which, maxiter, cvg_crit, k, is_V, ncv, verbose);
      }

      cytnx_error_msg(cvg_crit < 0, "[ERROR][Lanczos] cvg_crit should be >= 0%s", "\n");
      cytnx_error_msg((ncv != 0) && ((ncv < 2 + k) || ncv > Hop->nx()),
                      "[ERROR][Lanczos] ncv should "
                      "be 2+k<=ncv<=nx%s",
                      "\n");
      cytnx_uint64 output_size = is_V ? 2 : 1;
      auto out = std::vector<Tensor>(output_size, Tensor());
      KrylovStats stats;
      stats.algorithm = "Lanczos";
      stats.maxiter_requested = maxiter;
      stats.cvgcrit_requested = cvg_crit;
      stats.cvgcrit_used = cvg_crit;
      stats.input_dtype = T_init.dtype();
      stats.op_dtype = Hop->dtype();
      stats.working_dtype = _T_init.dtype();
      _Lanczos(out, Hop, _T_init, which, maxiter, cvg_crit, k, is_V, ncv, verbose, &stats);
      set_last_krylov_stats(stats);
      return out;
    }

    std::vector<UniTensor> Lanczos(LinOp *Hop, const UniTensor &UT_init, const std::string which,
                                   const cytnx_uint64 &maxiter, const double &cvg_crit,
                                   const cytnx_uint64 &k, const bool &is_V, const cytnx_int32 &ncv,
                                   const bool &verbose) {
      // check which
      std::vector<std::string> accept_which = {"LM", "LA", "SA"};
      if (std::find(accept_which.begin(), accept_which.end(), which) == accept_which.end()) {
        cytnx_error_msg(true, "[ERROR][Lanczos] 'which' should be 'LM', 'LA', 'SA'", "\n");
      }

      // If the operator is complex Hermitian, just call Arnoldi algorthm since there is no
      //     specific routine for complex Hermitian operaton in arpack.
      /// check k
      cytnx_error_msg(k < 1, "[ERROR][Lanczos] k should be >0%s", "\n");
      cytnx_error_msg(k > Hop->nx(),
                      "[ERROR][Lanczos] k can only be up to total dimension of input vector D%s",
                      "\n");

      // check Tin should be rank-1:
      auto _UT_init = UT_init;
      if (UT_init.dtype() == Type.Void) {
        cytnx_error_msg(k < 1, "[ERROR][Lanczos] The initial UniTensor sould be defined.%s", "\n");
      } else {
        int dim = get_elem_num(UT_init);
        cytnx_error_msg(dim != Hop->nx(),
                        "[ERROR][Lanczos] Tin should have dimension consistent with Hop: [%d] %s",
                        Hop->nx(), "\n");
        cytnx_error_msg(!Type.is_float(UT_init.dtype()),
                        "[ERROR][Lanczos] Tin should have floating dtype.%s", "\n");
        _UT_init = UT_init.astype(promoted_working_dtype(UT_init.dtype(), Hop->dtype()));
      }

      if (Type.is_complex(_UT_init.dtype())) {
        std::string tmp_which = which;
        if (which == "LA") {
          tmp_which = "LR";
        } else if (which == "SA") {
          tmp_which = "SR";
        }
        return Arnoldi(Hop, _UT_init, tmp_which, maxiter, cvg_crit, k, is_V, ncv, verbose);
      }

      cytnx_error_msg(cvg_crit < 0, "[ERROR][Lanczos] cvg_crit should be >= 0%s", "\n");
      cytnx_error_msg((ncv != 0) && ((ncv < 2 + k) || ncv > Hop->nx()),
                      "[ERROR][Lanczos] ncv should "
                      "be 2+k<=ncv<=nx%s",
                      "\n");
      auto out = std::vector<UniTensor>();
      KrylovStats stats;
      stats.algorithm = "Lanczos_Ut";
      stats.maxiter_requested = maxiter;
      stats.cvgcrit_requested = cvg_crit;
      stats.cvgcrit_used = cvg_crit;
      stats.input_dtype = UT_init.dtype();
      stats.op_dtype = Hop->dtype();
      stats.working_dtype = _UT_init.dtype();
      _Lanczos(out, Hop, _UT_init, which, maxiter, cvg_crit, k, is_V, ncv, verbose, &stats);
      set_last_krylov_stats(stats);
      return out;
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
