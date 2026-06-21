#include "linalg.hpp"
#include "Generator.hpp"
#include "LinOp.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "random.hpp"
#include "Lanczos_Gnd_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace linalg {
    namespace {

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

      unsigned int hermitian_projection_dtype_internal(const unsigned int dtype) {
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          return Type.Float;
        }
        return Type.Double;
      }

      Scalar real_scalar_for_dtype_internal(const double value, const unsigned int dtype) {
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          return Scalar(static_cast<cytnx_float>(value));
        }
        return Scalar(static_cast<cytnx_double>(value));
      }

      double scalar_to_double_internal(const Scalar &value) {
        return static_cast<double>(value.real());
      }

      double hermitian_projection_real_internal(const Scalar &value, const unsigned int dtype,
                                                const char *context) {
        const double real_part = scalar_to_double_internal(value.real());
        const double imag_part =
          Type.is_complex(value.dtype()) ? scalar_to_double_internal(value.imag()) : 0.0;
        const double scale = std::max({std::abs(real_part), std::abs(imag_part), 1.0});
        const double tolerance = 1000.0 * internal::dtype_epsilon(dtype) * scale;
        cytnx_error_msg(std::abs(imag_part) > tolerance,
                        "[ERROR][Lanczos_Gnd] %s produced a non-real Hermitian projection: "
                        "real=%.17g imag=%.17g tolerance=%.17g.%s",
                        context, real_part, imag_part, tolerance, "\n");
        return real_part;
      }

      Tensor eigenvalue_tensor_internal(const double value, const unsigned int working_dtype,
                                        const int device) {
        const unsigned int output_dtype = hermitian_projection_dtype_internal(working_dtype);
        Tensor out = zeros({1}, output_dtype, device);
        out(0) = real_scalar_for_dtype_internal(value, output_dtype);
        return out;
      }

      Scalar unitensor_dot_internal(const UniTensor &left, const UniTensor &right) {
        if (left.uten_type() == UTenType.BlockFermionic) {
          return Contract(left.Dagger().fermion_twists(), right).item();
        }
        return Contract(left.Dagger(), right).item();
      }

      struct TensorLanczosOps {
        using vector_type = Tensor;

        explicit TensorLanczosOps(LinOp *Hop, const unsigned int working_dtype)
            : Hop(Hop), working_dtype_(working_dtype) {}

        std::uint64_t dimension() const { return Hop->nx(); }
        unsigned int working_dtype() const { return working_dtype_; }

        Tensor matvec(const Tensor &v) const { return Hop->matvec(v).astype(working_dtype_); }

        void check_matvec_output(const Tensor &out, const Tensor &reference) const {
          cytnx_error_msg(out.shape().size() != 1,
                          "[ERROR][Lanczos_Gnd] LinOp.matvec(Tensor) must return a rank-1 "
                          "Tensor.%s",
                          "\n");
          cytnx_error_msg(out.shape()[0] != Hop->nx(),
                          "[ERROR][Lanczos_Gnd] LinOp.matvec(Tensor) returned dimension %llu but "
                          "LinOp::nx() is %llu.%s",
                          static_cast<unsigned long long>(out.shape()[0]),
                          static_cast<unsigned long long>(Hop->nx()), "\n");
          cytnx_error_msg(out.shape() != reference.shape(),
                          "[ERROR][Lanczos_Gnd] LinOp.matvec(Tensor) changed the vector shape.%s",
                          "\n");
        }

        double norm(const Tensor &v) const { return scalar_to_double_internal(v.Norm().item()); }

        double hermitian_inner_product_real(const Tensor &left, const Tensor &right) const {
          return hermitian_projection_real_internal(linalg::Vectordot(left, right, true).item(),
                                                    working_dtype_, "Tensor inner product");
        }

        Tensor scale(const Tensor &v, const double factor) const {
          return real_scalar_for_dtype_internal(factor, working_dtype_) * v;
        }

        void axpy(Tensor &y, const double alpha, const Tensor &x) const {
          y += real_scalar_for_dtype_internal(alpha, working_dtype_) * x;
        }

        LinOp *Hop;
        unsigned int working_dtype_;
      };

      struct UniTensorLanczosOps {
        using vector_type = UniTensor;

        explicit UniTensorLanczosOps(LinOp *Hop, const unsigned int working_dtype)
            : Hop(Hop), working_dtype_(working_dtype) {}

        std::uint64_t dimension() const { return Hop->nx(); }
        unsigned int working_dtype() const { return working_dtype_; }

        UniTensor matvec(const UniTensor &v) const {
          UniTensor out = Hop->matvec(v).astype(working_dtype_);
          out.apply_();
          return out;
        }

        void check_matvec_output(const UniTensor &out, const UniTensor &reference) const {
          cytnx_error_msg(out.labels().size() != reference.labels().size(),
                          "[ERROR][Lanczos_Gnd] LinOp.matvec(UniTensor) output must have the same "
                          "labels and shape as input.%s",
                          "\n");
          cytnx_error_msg(out.labels() != reference.labels(),
                          "[ERROR][Lanczos_Gnd] LinOp.matvec(UniTensor) output must have the same "
                          "labels as input.%s",
                          "\n");
          cytnx_error_msg(out.shape() != reference.shape(),
                          "[ERROR][Lanczos_Gnd] LinOp.matvec(UniTensor) output must have the same "
                          "shape as input.%s",
                          "\n");
        }

        double norm(const UniTensor &v) const { return scalar_to_double_internal(v.Norm().item()); }

        double hermitian_inner_product_real(const UniTensor &left, const UniTensor &right) const {
          return hermitian_projection_real_internal(unitensor_dot_internal(left, right),
                                                    working_dtype_, "UniTensor inner product");
        }

        UniTensor scale(const UniTensor &v, const double factor) const {
          return real_scalar_for_dtype_internal(factor, working_dtype_) * v;
        }

        void axpy(UniTensor &y, const double alpha, const UniTensor &x) const {
          y += real_scalar_for_dtype_internal(alpha, working_dtype_) * x;
        }

        LinOp *Hop;
        unsigned int working_dtype_;
      };

      void initialize_common_stats(KrylovStats &stats, const std::string &algorithm,
                                   const unsigned int maxiter, const double residual_tol,
                                   const unsigned int input_dtype,
                                   const unsigned int working_dtype) {
        stats.algorithm = algorithm;
        stats.maxiter_requested = maxiter;
        stats.residual_tol_requested = residual_tol;
        stats.residual_tol_used = residual_tol;
        stats.input_dtype = input_dtype;
        stats.working_dtype = working_dtype;
      }

    }  // namespace

    std::vector<Tensor> Lanczos_Gnd(LinOp *Hop, double residual_tol, bool is_V, const Tensor &Tin,
                                    bool verbose, unsigned int Maxiter) {
      cytnx_error_msg(Maxiter < 1, "[ERROR][Lanczos_Gnd] Maxiter must be at least 1.%s", "\n");

      const unsigned int working_dtype = promoted_working_dtype_internal(Tin.dtype(), Hop->dtype());
      Tensor v0;
      if (Tin.dtype() == Type.Void) {
        v0 = cytnx::random::normal({Hop->nx()}, 0, 1, Hop->device()).astype(working_dtype);
      } else {
        cytnx_error_msg(Tin.shape().size() != 1, "[ERROR][Lanczos_Gnd] Tin should be rank-1.%s",
                        "\n");
        cytnx_error_msg(Tin.shape()[0] != Hop->nx(),
                        "[ERROR][Lanczos_Gnd] Tin dimension %llu does not match LinOp::nx() "
                        "%llu.%s",
                        static_cast<unsigned long long>(Tin.shape()[0]),
                        static_cast<unsigned long long>(Hop->nx()), "\n");
        cytnx_error_msg(!Type.is_float(Tin.dtype()),
                        "[ERROR][Lanczos_Gnd] input tensors must have real or complex floating "
                        "dtype.%s",
                        "\n");
        v0 = Tin.astype(working_dtype);
      }

      TensorLanczosOps ops(Hop, working_dtype);
      KrylovStats stats;
      initialize_common_stats(stats, "Lanczos_Gnd", Maxiter, residual_tol, Tin.dtype(),
                              working_dtype);
      auto result =
        internal::lanczos_ground_state(ops, v0, is_V, residual_tol, Maxiter, verbose, &stats);
      set_last_krylov_stats(stats);

      std::vector<Tensor> out;
      out.push_back(eigenvalue_tensor_internal(result.eigenvalue, working_dtype, v0.device()));
      if (is_V) {
        out.push_back(result.eigenvector);
      }
      return out;
    }

    std::vector<UniTensor> Lanczos_Gnd_Ut(LinOp *Hop, const UniTensor &Tin, double residual_tol,
                                          bool is_V, bool verbose, unsigned int Maxiter) {
      cytnx_error_msg(Maxiter < 1, "[ERROR][Lanczos_Gnd] Maxiter must be at least 1.%s", "\n");
      cytnx_error_msg(!Type.is_float(Tin.dtype()),
                      "[ERROR][Lanczos_Gnd] input tensors must have real or complex floating "
                      "dtype.%s",
                      "\n");

      const unsigned int working_dtype = promoted_working_dtype_internal(Tin.dtype(), Hop->dtype());
      UniTensor v0 = Tin.astype(working_dtype);
      v0.contiguous_();
      v0.apply_();

      UniTensorLanczosOps ops(Hop, working_dtype);
      KrylovStats stats;
      initialize_common_stats(stats, "Lanczos_Gnd_Ut", Maxiter, residual_tol, Tin.dtype(),
                              working_dtype);
      auto result =
        internal::lanczos_ground_state(ops, v0, is_V, residual_tol, Maxiter, verbose, &stats);
      set_last_krylov_stats(stats);

      std::vector<UniTensor> out;
      out.push_back(UniTensor(
        eigenvalue_tensor_internal(result.eigenvalue, working_dtype, Tin.device()), false, 0));
      if (is_V) {
        out.push_back(result.eigenvector);
      }
      return out;
    }

  }  // namespace linalg
}  // namespace cytnx

#endif
