#include "linalg.hpp"
#include "Generator.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"
#include "DenseMatrix_internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

using namespace std;
namespace cytnx {
  namespace linalg {

    namespace {
      using internal::DenseMatrix;

      template <typename Scalar>
      struct is_complex_scalar : std::false_type {};

      template <typename Real>
      struct is_complex_scalar<std::complex<Real>> : std::true_type {};

      template <typename Scalar>
      constexpr bool is_complex_scalar_v = is_complex_scalar<Scalar>::value;

      template <typename Scalar>
      using real_scalar_t = typename std::conditional_t<is_complex_scalar_v<Scalar>, Scalar,
                                                        std::complex<Scalar>>::value_type;

      template <typename Scalar>
      Scalar make_scalar(const cytnx_complex128 &value) {
        if constexpr (is_complex_scalar_v<Scalar>) {
          return Scalar(static_cast<real_scalar_t<Scalar>>(value.real()),
                        static_cast<real_scalar_t<Scalar>>(value.imag()));
        } else {
          cytnx_error_msg(value.imag() != 0.0,
                          "[ExpM] internal error, cannot convert complex input to real output.%s",
                          "\n");
          return static_cast<Scalar>(value.real());
        }
      }

      template <typename Scalar>
      Scalar make_scalar(const cytnx_complex64 &value) {
        if constexpr (is_complex_scalar_v<Scalar>) {
          return Scalar(static_cast<real_scalar_t<Scalar>>(value.real()),
                        static_cast<real_scalar_t<Scalar>>(value.imag()));
        } else {
          cytnx_error_msg(value.imag() != 0.0f,
                          "[ExpM] internal error, cannot convert complex input to real output.%s",
                          "\n");
          return static_cast<Scalar>(value.real());
        }
      }

      template <typename Scalar, typename T>
      Scalar make_scalar(const T &value) {
        if constexpr (is_complex_scalar_v<Scalar>) {
          return Scalar(static_cast<real_scalar_t<Scalar>>(value));
        } else {
          return static_cast<Scalar>(value);
        }
      }

      template <typename Scalar>
      Scalar tensor_value(const Tensor &tensor, const cytnx_uint64 row, const cytnx_uint64 col) {
        if (tensor.dtype() == Type.ComplexDouble) {
          return make_scalar<Scalar>(tensor.at<cytnx_complex128>({row, col}));
        } else if (tensor.dtype() == Type.ComplexFloat) {
          return make_scalar<Scalar>(tensor.at<cytnx_complex64>({row, col}));
        } else if (tensor.dtype() == Type.Double) {
          return make_scalar<Scalar>(tensor.at<cytnx_double>({row, col}));
        } else if (tensor.dtype() == Type.Float) {
          return make_scalar<Scalar>(tensor.at<cytnx_float>({row, col}));
        } else if (tensor.dtype() == Type.Uint64) {
          return make_scalar<Scalar>(tensor.at<cytnx_uint64>({row, col}));
        } else if (tensor.dtype() == Type.Int64) {
          return make_scalar<Scalar>(tensor.at<cytnx_int64>({row, col}));
        } else if (tensor.dtype() == Type.Uint32) {
          return make_scalar<Scalar>(tensor.at<cytnx_uint32>({row, col}));
        } else if (tensor.dtype() == Type.Int32) {
          return make_scalar<Scalar>(tensor.at<cytnx_int32>({row, col}));
        } else if (tensor.dtype() == Type.Uint16) {
          return make_scalar<Scalar>(tensor.at<cytnx_uint16>({row, col}));
        } else if (tensor.dtype() == Type.Int16) {
          return make_scalar<Scalar>(tensor.at<cytnx_int16>({row, col}));
        } else if (tensor.dtype() == Type.Bool) {
          return make_scalar<Scalar>(tensor.at<cytnx_bool>({row, col}));
        }
        cytnx_error_msg(true, "[ExpM] unsupported input dtype.%s", "\n");
        return Scalar{};
      }

      template <typename Scalar>
      DenseMatrix<Scalar> identity_matrix(const cytnx_uint64 n) {
        DenseMatrix<Scalar> result(n, n);
        for (cytnx_uint64 i = 0; i < n; ++i) result(i, i) = Scalar{1};
        return result;
      }

      template <typename Scalar>
      DenseMatrix<Scalar> add(const DenseMatrix<Scalar> &lhs, const DenseMatrix<Scalar> &rhs) {
        DenseMatrix<Scalar> result(lhs.rows(), lhs.cols());
        for (cytnx_uint64 i = 0; i < lhs.rows(); ++i) {
          for (cytnx_uint64 j = 0; j < lhs.cols(); ++j) result(i, j) = lhs(i, j) + rhs(i, j);
        }
        return result;
      }

      template <typename Scalar>
      DenseMatrix<Scalar> subtract(const DenseMatrix<Scalar> &lhs, const DenseMatrix<Scalar> &rhs) {
        DenseMatrix<Scalar> result(lhs.rows(), lhs.cols());
        for (cytnx_uint64 i = 0; i < lhs.rows(); ++i) {
          for (cytnx_uint64 j = 0; j < lhs.cols(); ++j) result(i, j) = lhs(i, j) - rhs(i, j);
        }
        return result;
      }

      template <typename Scalar>
      DenseMatrix<Scalar> scale(const DenseMatrix<Scalar> &mat, const Scalar &scalar) {
        DenseMatrix<Scalar> result(mat.rows(), mat.cols());
        for (cytnx_uint64 i = 0; i < mat.rows(); ++i) {
          for (cytnx_uint64 j = 0; j < mat.cols(); ++j) result(i, j) = mat(i, j) * scalar;
        }
        return result;
      }

      template <typename Scalar>
      DenseMatrix<Scalar> multiply(const DenseMatrix<Scalar> &lhs, const DenseMatrix<Scalar> &rhs) {
        DenseMatrix<Scalar> result(lhs.rows(), rhs.cols());
        for (cytnx_uint64 i = 0; i < lhs.rows(); ++i) {
          for (cytnx_uint64 k = 0; k < lhs.cols(); ++k) {
            const Scalar factor = lhs(i, k);
            if (factor == Scalar{}) continue;
            for (cytnx_uint64 j = 0; j < rhs.cols(); ++j) result(i, j) += factor * rhs(k, j);
          }
        }
        return result;
      }

      template <typename Scalar>
      real_scalar_t<Scalar> matrix_one_norm(const DenseMatrix<Scalar> &mat) {
        real_scalar_t<Scalar> result{};
        for (cytnx_uint64 j = 0; j < mat.cols(); ++j) {
          real_scalar_t<Scalar> column_sum{};
          for (cytnx_uint64 i = 0; i < mat.rows(); ++i) {
            column_sum += std::abs(mat(i, j));
          }
          result = std::max(result, column_sum);
        }
        return result;
      }

      template <typename Scalar>
      DenseMatrix<Scalar> solve_linear_system(DenseMatrix<Scalar> a, DenseMatrix<Scalar> b) {
        const cytnx_uint64 n = a.rows();
        const cytnx_uint64 nrhs = b.cols();

        for (cytnx_uint64 k = 0; k < n; ++k) {
          cytnx_uint64 pivot_row = k;
          real_scalar_t<Scalar> pivot_value = std::abs(a(k, k));
          for (cytnx_uint64 i = k + 1; i < n; ++i) {
            const real_scalar_t<Scalar> candidate = std::abs(a(i, k));
            if (candidate > pivot_value) {
              pivot_value = candidate;
              pivot_row = i;
            }
          }

          cytnx_error_msg(pivot_value == real_scalar_t<Scalar>{},
                          "[ExpM] Padé denominator is numerically singular.%s", "\n");

          if (pivot_row != k) {
            for (cytnx_uint64 j = 0; j < n; ++j) std::swap(a(k, j), a(pivot_row, j));
            for (cytnx_uint64 j = 0; j < nrhs; ++j) std::swap(b(k, j), b(pivot_row, j));
          }

          const Scalar pivot = a(k, k);
          for (cytnx_uint64 i = k + 1; i < n; ++i) {
            const Scalar factor = a(i, k) / pivot;
            if (factor == Scalar{}) continue;
            a(i, k) = Scalar{};
            for (cytnx_uint64 j = k + 1; j < n; ++j) a(i, j) -= factor * a(k, j);
            for (cytnx_uint64 j = 0; j < nrhs; ++j) b(i, j) -= factor * b(k, j);
          }
        }

        DenseMatrix<Scalar> x = b;
        for (cytnx_int64 ii = static_cast<cytnx_int64>(n) - 1; ii >= 0; --ii) {
          const cytnx_uint64 i = static_cast<cytnx_uint64>(ii);
          const Scalar pivot = a(i, i);
          for (cytnx_uint64 j = 0; j < nrhs; ++j) {
            Scalar value = x(i, j);
            for (cytnx_uint64 k = i + 1; k < n; ++k) value -= a(i, k) * x(k, j);
            x(i, j) = value / pivot;
          }
        }
        return x;
      }

      template <typename Scalar>
      DenseMatrix<Scalar> linear_combination(
        const cytnx_uint64 n,
        const std::initializer_list<std::pair<const DenseMatrix<Scalar> *, Scalar>> terms) {
        DenseMatrix<Scalar> result(n, n);
        for (const auto &term : terms) {
          for (cytnx_uint64 i = 0; i < n; ++i) {
            for (cytnx_uint64 j = 0; j < n; ++j) result(i, j) += term.second * (*term.first)(i, j);
          }
        }
        return result;
      }

      template <typename Scalar>
      DenseMatrix<Scalar> solve_pade(const DenseMatrix<Scalar> &u, const DenseMatrix<Scalar> &v) {
        return solve_linear_system(subtract(v, u), add(v, u));
      }

      template <typename Scalar>
      Scalar from_double(const double value) {
        return Scalar(static_cast<real_scalar_t<Scalar>>(value));
      }

      template <typename Scalar>
      DenseMatrix<Scalar> pade3(const DenseMatrix<Scalar> &a) {
        static constexpr std::array<double, 4> b{120.0, 60.0, 12.0, 1.0};
        const cytnx_uint64 n = a.rows();
        const DenseMatrix<Scalar> ident = identity_matrix<Scalar>(n);
        const DenseMatrix<Scalar> a2 = multiply(a, a);
        const DenseMatrix<Scalar> u = multiply(
          a, linear_combination<Scalar>(
               n, {{&a2, from_double<Scalar>(b[3])}, {&ident, from_double<Scalar>(b[1])}}));
        const DenseMatrix<Scalar> v = linear_combination<Scalar>(
          n, {{&a2, from_double<Scalar>(b[2])}, {&ident, from_double<Scalar>(b[0])}});
        return solve_pade(u, v);
      }

      template <typename Scalar>
      DenseMatrix<Scalar> pade5(const DenseMatrix<Scalar> &a) {
        static constexpr std::array<double, 6> b{30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0};
        const cytnx_uint64 n = a.rows();
        const DenseMatrix<Scalar> ident = identity_matrix<Scalar>(n);
        const DenseMatrix<Scalar> a2 = multiply(a, a);
        const DenseMatrix<Scalar> a4 = multiply(a2, a2);
        const DenseMatrix<Scalar> u =
          multiply(a, linear_combination<Scalar>(n, {{&a4, from_double<Scalar>(b[5])},
                                                     {&a2, from_double<Scalar>(b[3])},
                                                     {&ident, from_double<Scalar>(b[1])}}));
        const DenseMatrix<Scalar> v =
          linear_combination<Scalar>(n, {{&a4, from_double<Scalar>(b[4])},
                                         {&a2, from_double<Scalar>(b[2])},
                                         {&ident, from_double<Scalar>(b[0])}});
        return solve_pade(u, v);
      }

      template <typename Scalar>
      DenseMatrix<Scalar> pade7(const DenseMatrix<Scalar> &a) {
        static constexpr std::array<double, 8> b{17297280.0, 8648640.0, 1995840.0, 277200.0,
                                                 25200.0,    1512.0,    56.0,      1.0};
        const cytnx_uint64 n = a.rows();
        const DenseMatrix<Scalar> ident = identity_matrix<Scalar>(n);
        const DenseMatrix<Scalar> a2 = multiply(a, a);
        const DenseMatrix<Scalar> a4 = multiply(a2, a2);
        const DenseMatrix<Scalar> a6 = multiply(a4, a2);
        const DenseMatrix<Scalar> u =
          multiply(a, linear_combination<Scalar>(n, {{&a6, from_double<Scalar>(b[7])},
                                                     {&a4, from_double<Scalar>(b[5])},
                                                     {&a2, from_double<Scalar>(b[3])},
                                                     {&ident, from_double<Scalar>(b[1])}}));
        const DenseMatrix<Scalar> v =
          linear_combination<Scalar>(n, {{&a6, from_double<Scalar>(b[6])},
                                         {&a4, from_double<Scalar>(b[4])},
                                         {&a2, from_double<Scalar>(b[2])},
                                         {&ident, from_double<Scalar>(b[0])}});
        return solve_pade(u, v);
      }

      template <typename Scalar>
      DenseMatrix<Scalar> pade9(const DenseMatrix<Scalar> &a) {
        static constexpr std::array<double, 10> b{
          17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
          2162160.0,     110880.0,     3960.0,       90.0,        1.0};
        const cytnx_uint64 n = a.rows();
        const DenseMatrix<Scalar> ident = identity_matrix<Scalar>(n);
        const DenseMatrix<Scalar> a2 = multiply(a, a);
        const DenseMatrix<Scalar> a4 = multiply(a2, a2);
        const DenseMatrix<Scalar> a6 = multiply(a4, a2);
        const DenseMatrix<Scalar> a8 = multiply(a6, a2);
        const DenseMatrix<Scalar> u =
          multiply(a, linear_combination<Scalar>(n, {{&a8, from_double<Scalar>(b[9])},
                                                     {&a6, from_double<Scalar>(b[7])},
                                                     {&a4, from_double<Scalar>(b[5])},
                                                     {&a2, from_double<Scalar>(b[3])},
                                                     {&ident, from_double<Scalar>(b[1])}}));
        const DenseMatrix<Scalar> v =
          linear_combination<Scalar>(n, {{&a8, from_double<Scalar>(b[8])},
                                         {&a6, from_double<Scalar>(b[6])},
                                         {&a4, from_double<Scalar>(b[4])},
                                         {&a2, from_double<Scalar>(b[2])},
                                         {&ident, from_double<Scalar>(b[0])}});
        return solve_pade(u, v);
      }

      template <typename Scalar>
      DenseMatrix<Scalar> pade13(const DenseMatrix<Scalar> &a, const DenseMatrix<Scalar> &a2,
                                 const DenseMatrix<Scalar> &a4, const DenseMatrix<Scalar> &a6) {
        static constexpr std::array<double, 14> b{64764752532480000.0,
                                                  32382376266240000.0,
                                                  7771770303897600.0,
                                                  1187353796428800.0,
                                                  129060195264000.0,
                                                  10559470521600.0,
                                                  670442572800.0,
                                                  33522128640.0,
                                                  1323241920.0,
                                                  40840800.0,
                                                  960960.0,
                                                  16380.0,
                                                  182.0,
                                                  1.0};
        const cytnx_uint64 n = a.rows();
        const DenseMatrix<Scalar> ident = identity_matrix<Scalar>(n);
        const DenseMatrix<Scalar> first =
          linear_combination<Scalar>(n, {{&a6, from_double<Scalar>(b[13])},
                                         {&a4, from_double<Scalar>(b[11])},
                                         {&a2, from_double<Scalar>(b[9])}});
        DenseMatrix<Scalar> tmp = multiply(a6, first);
        tmp = add(tmp, linear_combination<Scalar>(n, {{&a6, from_double<Scalar>(b[7])},
                                                      {&a4, from_double<Scalar>(b[5])},
                                                      {&a2, from_double<Scalar>(b[3])},
                                                      {&ident, from_double<Scalar>(b[1])}}));
        const DenseMatrix<Scalar> u = multiply(a, tmp);

        const DenseMatrix<Scalar> third =
          linear_combination<Scalar>(n, {{&a6, from_double<Scalar>(b[12])},
                                         {&a4, from_double<Scalar>(b[10])},
                                         {&a2, from_double<Scalar>(b[8])}});
        DenseMatrix<Scalar> v = multiply(a6, third);
        v = add(v, linear_combination<Scalar>(n, {{&a6, from_double<Scalar>(b[6])},
                                                  {&a4, from_double<Scalar>(b[4])},
                                                  {&a2, from_double<Scalar>(b[2])},
                                                  {&ident, from_double<Scalar>(b[0])}}));
        return solve_pade(u, v);
      }

      template <typename Scalar>
      DenseMatrix<Scalar> matrix_exponential(DenseMatrix<Scalar> a) {
        // Scaling-and-squaring with Padé orders and theta bounds from Higham (2005) and
        // Al-Mohy & Higham (2011). This avoids the previous Eig + InvM(eigenvectors) path, which
        // fails for defective matrices and is numerically fragile for non-normal matrices.
        static constexpr std::array<double, 5> theta{1.495585217958292e-2, 2.539398330063230e-1,
                                                     9.504178996162932e-1, 2.097847961257068,
                                                     5.371920351148152};

        const cytnx_uint64 n = a.rows();
        const real_scalar_t<Scalar> norm = matrix_one_norm(a);
        cytnx_error_msg(!std::isfinite(norm), "[ExpM] input matrix must have finite norm.%s", "\n");
        if (norm == real_scalar_t<Scalar>{}) return identity_matrix<Scalar>(n);

        if (norm <= theta[0]) return pade3(a);
        if (norm <= theta[1]) return pade5(a);
        if (norm <= theta[2]) return pade7(a);
        if (norm <= theta[3]) return pade9(a);

        int scaling = 0;
        if (norm > theta[4]) {
          scaling = static_cast<int>(std::ceil(std::log2(norm / theta[4])));
          a = scale(a, from_double<Scalar>(std::ldexp(1.0, -scaling)));
        }

        const DenseMatrix<Scalar> a2 = multiply(a, a);
        const DenseMatrix<Scalar> a4 = multiply(a2, a2);
        const DenseMatrix<Scalar> a6 = multiply(a4, a2);
        DenseMatrix<Scalar> result = pade13(a, a2, a4, a6);
        for (int i = 0; i < scaling; ++i) result = multiply(result, result);
        return result;
      }

      template <typename Scalar>
      Tensor matrix_to_tensor(const DenseMatrix<Scalar> &matrix, const unsigned int dtype,
                              const int device) {
        Tensor out({matrix.rows(), matrix.cols()}, dtype, device);
        for (cytnx_uint64 i = 0; i < matrix.rows(); ++i) {
          for (cytnx_uint64 j = 0; j < matrix.cols(); ++j) {
            if constexpr (std::is_same_v<Scalar, cytnx_complex128>) {
              out.at<cytnx_complex128>({i, j}) = matrix(i, j);
            } else if constexpr (std::is_same_v<Scalar, cytnx_complex64>) {
              out.at<cytnx_complex64>({i, j}) = matrix(i, j);
            } else if constexpr (std::is_same_v<Scalar, cytnx_double>) {
              out.at<cytnx_double>({i, j}) = matrix(i, j);
            } else if constexpr (std::is_same_v<Scalar, cytnx_float>) {
              out.at<cytnx_float>({i, j}) = matrix(i, j);
            }
          }
        }
        return out;
      }

      template <typename Scalar, typename T>
      Tensor expm_impl(const Tensor &Tin, const T &a, const T &b, const unsigned int dtype) {
        const cytnx_uint64 n = Tin.shape()[0];
        DenseMatrix<Scalar> matrix(n, n);
        const Scalar aa = make_scalar<Scalar>(a);
        const Scalar bb = make_scalar<Scalar>(b);
        for (cytnx_uint64 i = 0; i < n; ++i) {
          for (cytnx_uint64 j = 0; j < n; ++j) {
            matrix(i, j) = aa * tensor_value<Scalar>(Tin, i, j);
            if (i == j) matrix(i, j) += bb;
          }
        }
        return matrix_to_tensor(matrix_exponential(matrix), dtype, Tin.device());
      }

      template <typename T>
      bool scalar_is_complex() {
        return std::is_same_v<T, cytnx_complex128> || std::is_same_v<T, cytnx_complex64>;
      }

    }  // namespace

    template <typename T>
    Tensor ExpM(const Tensor &Tin, const T &a, const T &b) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[ExpM] error, ExpM can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[ExpM] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[ExpM] error, ExpM can only operator on square Tensor (#row = #col%s", "\n");

      cytnx_error_msg(Tin.device() != Device.cpu,
                      "[ExpM] GPU matrix exponential is not implemented.%s", "\n");
      cytnx_error_msg(Tin.dtype() == Type.Void, "[ExpM] input Tensor has Void dtype.%s", "\n");

      const bool input_complex = Type.is_complex(Tin.dtype());
      const bool output_complex = input_complex || scalar_is_complex<T>();
      const bool output_double = Tin.dtype() != Type.Float && Tin.dtype() != Type.ComplexFloat;

      if (output_complex) {
        if (output_double) return expm_impl<cytnx_complex128>(Tin, a, b, Type.ComplexDouble);
        return expm_impl<cytnx_complex64>(Tin, a, b, Type.ComplexFloat);
      }
      if (output_double) return expm_impl<cytnx_double>(Tin, a, b, Type.Double);
      return expm_impl<cytnx_float>(Tin, a, b, Type.Float);
    }
    Tensor ExpM(const Tensor &Tin) { return linalg::ExpM(Tin, double(1), double(0)); }
    template Tensor ExpM(const Tensor &Tin, const cytnx_complex128 &a, const cytnx_complex128 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_complex64 &a, const cytnx_complex64 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_double &a, const cytnx_double &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_float &a, const cytnx_float &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_uint64 &a, const cytnx_uint64 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_uint32 &a, const cytnx_uint32 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_uint16 &a, const cytnx_uint16 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_int32 &a, const cytnx_int32 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_int16 &a, const cytnx_int16 &b);

  }  // namespace linalg

}  // namespace cytnx

namespace cytnx {
  namespace linalg {

    template <typename T>
    static void ExpM_Dense_UT_internal(UniTensor &out, const UniTensor &Tin, const T &a,
                                       const T &b) {
      cytnx_int64 Drow = 1, Dcol = 1;
      for (int i = 0; i < Tin.rowrank(); i++) {
        Drow *= Tin.shape()[i];
      }
      for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
        Dcol *= Tin.shape()[i];
      }
      cytnx_error_msg(
        Drow != Dcol,
        "[ERROR][ExpM] The total dimension of row-space and col-space should be equal!!%s", "\n");

      out.get_block_().reshape_({Drow, Dcol});

      out.get_block_() = cytnx::linalg::ExpM(out.get_block_(), a, b);

      out.get_block_().reshape_(Tin.shape());
    }

    template <typename T>
    static void ExpM_Sparse_UT_internal(UniTensor &out, const UniTensor &Tin, const T &a,
                                        const T &b) {
      std::vector<Tensor> &tmp = out.get_blocks_();

      for (int i = 0; i < tmp.size(); i++) {
        tmp[i] = cytnx::linalg::ExpM(tmp[i], a, b);
      }
    }

    // Block-wise matrix exponential of a general symmetric UniTensor. Handles both BlockUniTensor
    // (bosonic) and BlockFermionicUniTensor (fermionic), selected by the template parameter BUT.
    // For the fermionic case, sign-flipped blocks are negated to the physical operator before each
    // per-qcharge dense ExpM, and the result is stored with an all-false signflip (it is already
    // physical). For the bosonic case there are no sign flips and those steps do nothing.
    template <class BUT, typename T>
    static void ExpM_BlockUT_internal(UniTensor &out, const UniTensor &Tin, const T &a,
                                      const T &b) {
      std::vector<bool> signflip;
      if constexpr (std::is_same_v<BUT, BlockFermionicUniTensor>)
        signflip = static_cast<BlockFermionicUniTensor *>(Tin._impl.get())->_signflip;

      // 1) getting the combineBond L and combineBond R for qnum list without grouping:
      //
      //   BDLeft -[ ]- BDRight
      //
      std::vector<cytnx_uint64> strides;
      strides.reserve(Tin.rank());
      auto BdLeft = Tin.bonds()[0].clone();
      for (int i = 1; i < Tin.rowrank(); i++) {
        strides.push_back(Tin.bonds()[i].qnums().size());
        BdLeft._impl->force_combineBond_(Tin.bonds()[i]._impl, false);  // no grouping
      }
      strides.push_back(1);
      auto BdRight = Tin.bonds()[Tin.rowrank()].clone();
      for (int i = Tin.rowrank() + 1; i < Tin.rank(); i++) {
        strides.push_back(Tin.bonds()[i].qnums().size());
        BdRight._impl->force_combineBond_(Tin.bonds()[i]._impl, false);  // no grouping
      }
      strides.push_back(1);

      // 2) making new inner_to_outer_idx lists for each block:
      // -> a. get stride:
      for (int i = Tin.rowrank() - 2; i >= 0; i--) {
        strides[i] *= strides[i + 1];
      }
      for (int i = Tin.rank() - 2; i >= Tin.rowrank(); i--) {
        strides[i] *= strides[i + 1];
      }
      //  ->b. calc new inner_to_outer_idx!
      vec2d<cytnx_uint64> new_itoi(Tin.Nblocks(), std::vector<cytnx_uint64>(2));

      int cnt;
      for (cytnx_uint64 b = 0; b < Tin.Nblocks(); b++) {
        const std::vector<cytnx_uint64> &tmpv = Tin.get_qindices(b);
        for (cnt = 0; cnt < Tin.rowrank(); cnt++) {
          new_itoi[b][0] += tmpv[cnt] * strides[cnt];
        }
        for (cnt = Tin.rowrank(); cnt < Tin.rank(); cnt++) {
          new_itoi[b][1] += tmpv[cnt] * strides[cnt];
        }
      }

      // 3) categorize:
      // key = qnum, val = list of block locations:
      std::map<std::vector<cytnx_int64>, std::vector<cytnx_int64>> mgrp;
      const auto left_qnums =
        (BdLeft.type() == bondType::BD_IN) ? BdLeft.qnums() : BdLeft.calc_reverse_qnums();
      for (cytnx_uint64 b = 0; b < Tin.Nblocks(); b++) {
        mgrp[left_qnums[new_itoi[b][0]]].push_back(b);
      }

      // 4) for each qcharge in key, combining the blocks into a big chunk!
      vec2d<cytnx_uint64> &ref_itoi = ((BUT *)out._impl.get())->_inner_to_outer_idx;
      std::vector<Tensor> &out_blocks_ = ((BUT *)out._impl.get())->_blocks;
      for (auto const &x : mgrp) {
        vec2d<cytnx_uint64> itoi_indicators(x.second.size());
        for (int i = 0; i < x.second.size(); i++) {
          itoi_indicators[i] = new_itoi[x.second[i]];
        }
        auto order = vec_sort(itoi_indicators, true);
        std::vector<Tensor> Tlist(itoi_indicators.size());
        cytnx_uint64 Rblk_dim = 0;
        cytnx_int64 tmp = -1;
        cytnx_int64 row_szs;
        std::vector<cytnx_uint64> rdims, cdims;  // this is used to split!
        vec2d<cytnx_uint64> old_shape(order.size());

        for (int i = 0; i < order.size(); i++) {
          cytnx_int64 current_block = x.second[order[i]];
          Tlist[i] = Tin.get_blocks()[current_block];
          row_szs = 1;
          old_shape[i] = Tlist[i].shape();
          for (int j = 0; j < Tin.rowrank(); j++) {
            row_szs *= Tlist[i].shape()[j];
          }
          bool flip = false;
          if constexpr (std::is_same_v<BUT, BlockFermionicUniTensor>)
            flip = signflip[current_block];
          if (flip) Tlist[i] = -Tlist[i];  // negate to the physical operator before exp
          Tlist[i] = Tlist[i].reshape({row_szs, -1});
          if (itoi_indicators[i][0] != tmp) {
            tmp = itoi_indicators[i][0];
            Rblk_dim++;
            rdims.push_back(row_szs);
          }
        }
        cytnx_error_msg(Tlist.size() % Rblk_dim, "[Internal ERROR] Tlist is not complete!%s", "\n");
        cytnx_uint64 Cblk_dim = Tlist.size() / Rblk_dim;
        for (int i = 0; i < Cblk_dim; i++) {
          cdims.push_back(Tlist[i].shape()[1]);
        }
        // BTen is the big block!!
        Tensor BTen = algo::_fx_Matric_combine(Tlist, Rblk_dim, Cblk_dim);

        BTen = cytnx::linalg::ExpM(BTen, a, b);

        Tlist.clear();
        algo::_fx_Matric_split(Tlist, BTen, rdims, cdims);

        // resize:
        for (int i = 0; i < Tlist.size(); i++) {
          Tlist[i].reshape_(old_shape[i]);
        }

        // put into new blocks:
        out_blocks_.insert(out_blocks_.end(), Tlist.begin(), Tlist.end());

        // rebuild itoi:
        for (int i = 0; i < order.size(); i++) {
          ref_itoi.push_back(Tin.get_qindices(x.second[order[i]]));
        }

      }  // for each qcharge

      // the sign was already included when creating the per-qcharge blocks, so the resulting
      // signflip is all false.
      if constexpr (std::is_same_v<BUT, BlockFermionicUniTensor>)
        ((BUT *)out._impl.get())->_signflip = std::vector<bool>(out_blocks_.size(), false);
    }

    template <typename T>
    UniTensor ExpM(const UniTensor &Tin, const T &a, const T &b) {
      cytnx_error_msg(Tin.rowrank() < 1,
                      "[ERROR][ExpM] Input UniTensor should have rowrank > 0, but rowrank is %d\n",
                      Tin.rowrank());
      cytnx_error_msg(Tin.rowrank() >= Tin.rank(),
                      "[ERROR][ExpM] Input UniTensor should have rowrank < rank, but rowrank is %d "
                      "and rank is %d\n",
                      Tin.rowrank(), Tin.rank());

      if (Tin.uten_type() == UTenType.Dense) {
        UniTensor out;
        if (Tin.is_contiguous()) {
          out = Tin.clone();
        } else {
          out = Tin.contiguous();
        }
        ExpM_Dense_UT_internal(out, Tin, a, b);
        return out;
      } else if (Tin.uten_type() == UTenType.Block) {
        // copy everything except _blocks and _inner_to_outer_idx
        BlockUniTensor *raw_out = ((BlockUniTensor *)Tin._impl.get())->clone_meta(false, true);
        UniTensor out;
        out._impl = boost::intrusive_ptr<UniTensor_base>(raw_out);
        ExpM_BlockUT_internal<BlockUniTensor>(out, Tin, a, b);
        return out;
      } else if (Tin.uten_type() == UTenType.BlockFermionic) {
        // copy everything except _blocks and _inner_to_outer_idx
        BlockFermionicUniTensor *raw_out =
          ((BlockFermionicUniTensor *)Tin._impl.get())->clone_meta(false, true);
        UniTensor out;
        out._impl = boost::intrusive_ptr<UniTensor_base>(raw_out);
        ExpM_BlockUT_internal<BlockFermionicUniTensor>(out, Tin, a, b);
        return out;
      } else if (Tin.uten_type() == UTenType.Sparse) {
        UniTensor out;
        if (Tin.is_contiguous())
          out = Tin.clone();
        else
          out = Tin.contiguous();
        ExpM_Sparse_UT_internal(out, Tin, a, b);
        return out;
      } else {
        cytnx_error_msg(true, "[ERROR][ExpM] UniTensor type '%s' not supported\n",
                        Tin.uten_type_str().c_str());
      }
    }

    UniTensor ExpM(const UniTensor &Tin) { return linalg::ExpM(Tin, double(1), double(0)); }

    template UniTensor ExpM(const UniTensor &Tin, const cytnx_complex128 &a,
                            const cytnx_complex128 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_complex64 &a,
                            const cytnx_complex64 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_double &a, const cytnx_double &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_float &a, const cytnx_float &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_uint16 &a, const cytnx_uint16 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_uint32 &a, const cytnx_uint32 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_uint64 &a, const cytnx_uint64 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_int16 &a, const cytnx_int16 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_int32 &a, const cytnx_int32 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b);

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
