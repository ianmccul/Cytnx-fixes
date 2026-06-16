#ifndef CYTNX_TENSORT_TRAITS_HPP_
#define CYTNX_TENSORT_TRAITS_HPP_

#include "TensorT.hpp"
#include "Type.hpp"
#include "mdspan_concepts.hpp"

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace cytnx {

  /// Element type concept for the supported real floating-point TensorT scalar types.
  template <class T>
  concept RealScalar = std::same_as<std::remove_cv_t<T>, cytnx_float> ||
    std::same_as<std::remove_cv_t<T>, cytnx_double>;

  /// Element type concept for the supported complex floating-point TensorT scalar types.
  template <class T>
  concept ComplexScalar = std::same_as<std::remove_cv_t<T>, cytnx_complex64> ||
    std::same_as<std::remove_cv_t<T>, cytnx_complex128>;

  /// Element type concept for supported real or complex floating-point TensorT scalar types.
  template <class T>
  concept NumericScalar = RealScalar<T> || ComplexScalar<T>;

  /// Type list used to build real TensorT dispatch variants.
  using RealScalars = std::tuple<cytnx_float, cytnx_double>;

  /// Type list used to build complex TensorT dispatch variants.
  using ComplexScalars = std::tuple<cytnx_complex64, cytnx_complex128>;

  /// Type list used to build real-or-complex TensorT dispatch variants.
  using NumericScalars = std::tuple<cytnx_float, cytnx_double, cytnx_complex64, cytnx_complex128>;

#ifdef UNI_GPU
  using TensorAccesses = std::tuple<host_access, cuda_access>;
#else
  using TensorAccesses = std::tuple<host_access>;
#endif

  namespace tensor_t_detail {

    template <std::size_t Rank, class Layout, class Scalar, class Accesses>
    struct tensor_variant_for_scalar;

    template <std::size_t Rank, class Layout, class Scalar, class... Accesses>
    struct tensor_variant_for_scalar<Rank, Layout, Scalar, std::tuple<Accesses...>> {
      using type = std::variant<TensorT<Scalar, Rank, Accesses, Layout>...>;
    };

    template <class... Variants>
    struct variant_cat;

    template <class... Types>
    struct variant_cat<std::variant<Types...>> {
      using type = std::variant<Types...>;
    };

    template <class... Left, class... Right, class... Rest>
    struct variant_cat<std::variant<Left...>, std::variant<Right...>, Rest...> {
      using type = typename variant_cat<std::variant<Left..., Right...>, Rest...>::type;
    };

    template <std::size_t Rank, class Layout, class Scalars, class Accesses>
    struct tensor_variant_from_lists;

    template <std::size_t Rank, class Layout, class... Scalars, class... Accesses>
    struct tensor_variant_from_lists<Rank, Layout, std::tuple<Scalars...>,
                                     std::tuple<Accesses...>> {
      using type = typename variant_cat<typename tensor_variant_for_scalar<
        Rank, Layout, Scalars, std::tuple<Accesses...>>::type...>::type;
    };

  }  // namespace tensor_t_detail

  template <std::size_t Rank, class Layout, class Scalars, class Accesses = TensorAccesses>
  using TensorVariantT =
    typename tensor_t_detail::tensor_variant_from_lists<Rank, Layout, Scalars, Accesses>::type;

  /// Variant over real scalar types and enabled access backends.
  template <std::size_t Rank, class Layout = stdex::layout_right>
  using RealTensor = TensorVariantT<Rank, Layout, RealScalars>;

  /// Variant over complex scalar types and enabled access backends.
  template <std::size_t Rank, class Layout = stdex::layout_right>
  using ComplexTensor = TensorVariantT<Rank, Layout, ComplexScalars>;

  /// Variant over real-or-complex scalar types and enabled access backends.
  template <std::size_t Rank, class Layout = stdex::layout_right>
  using NumericTensor = TensorVariantT<Rank, Layout, NumericScalars>;

  template <class T>
  struct is_variant : std::false_type {};

  template <class... Alternatives>
  struct is_variant<std::variant<Alternatives...>> : std::true_type {};

  /// True if `T` is a `std::variant`.
  template <class T>
  concept Variant = is_variant<std::remove_cvref_t<T>>::value;

  namespace tensor_t_detail {

    template <class T>
    struct dispatch_alternatives {
      using type = std::tuple<T>;
    };

    template <class... Alternatives>
    struct dispatch_alternatives<std::variant<Alternatives...>> {
      using type = std::tuple<Alternatives...>;
    };

    template <class T>
    using dispatch_alternatives_t = typename dispatch_alternatives<std::remove_cvref_t<T>>::type;

    template <class F, class Accumulated, class... AlternativeTuples>
    struct any_dispatch_invocable;

    template <class F, class... Accumulated>
    struct any_dispatch_invocable<F, std::tuple<Accumulated...>> {
      static constexpr bool value = std::is_invocable_v<F, Accumulated &...>;
    };

    template <class F, class... Accumulated, class... Alternatives, class... RestTuples>
    struct any_dispatch_invocable<F, std::tuple<Accumulated...>, std::tuple<Alternatives...>,
                                  RestTuples...> {
      static constexpr bool value =
        (any_dispatch_invocable<F, std::tuple<Accumulated..., Alternatives>,
                                RestTuples...>::value ||
         ...);
    };

    template <class F>
    decltype(auto) dispatch_visit(F &&f) {
      return std::forward<F>(f)();
    }

    template <class F, class Arg, class... Rest>
    decltype(auto) dispatch_visit(F &&f, Arg &&arg, Rest &&...rest) {
      if constexpr (Variant<Arg>) {
        return std::visit(
          [&](auto &&alternative) -> decltype(auto) {
            return dispatch_visit(
              [&](auto &&...tail) -> decltype(auto) {
                return std::forward<F>(f)(std::forward<decltype(alternative)>(alternative),
                                          std::forward<decltype(tail)>(tail)...);
              },
              std::forward<Rest>(rest)...);
          },
          std::forward<Arg>(arg));
      } else {
        return dispatch_visit(
          [&](auto &&...tail) -> decltype(auto) {
            return std::forward<F>(f)(std::forward<Arg>(arg),
                                      std::forward<decltype(tail)>(tail)...);
          },
          std::forward<Rest>(rest)...);
      }
    }

  }  // namespace tensor_t_detail

  /**
   * @brief True if at least one Cartesian product of dispatch alternatives is invocable by `F`.
   *
   * Each argument type may be a concrete type or a `std::variant`; non-variants are treated as
   * single-alternative dispatch arguments.
   */
  template <class F, class... Args>
  concept AnyDispatchInvocable = tensor_t_detail::any_dispatch_invocable<
    F, std::tuple<>, tensor_t_detail::dispatch_alternatives_t<Args>...>::value;

  namespace tensor_t_detail {

    template <class T>
    struct tensor_t_alternative;

    template <class T, std::size_t Rank, class Access, class Layout>
    struct tensor_t_alternative<TensorT<T, Rank, Access, Layout>> {
      using element_type = T;
      using access_type = Access;
      using layout_type = Layout;
      static constexpr std::size_t rank = Rank;
    };

    template <class Alternative>
    bool tensor_matches_alternative(const Tensor &tensor) {
      using traits = tensor_t_alternative<Alternative>;
      using element_type = typename traits::element_type;
      using access_type = typename traits::access_type;
      return tensor.rank() == traits::rank &&
             tensor.dtype() == Type_class::cy_typeid_v<std::remove_cv_t<element_type>> &&
             access_accepts_device(access_type{}, tensor.device());
    }

    template <class Alternative, class Variant>
    bool try_make_tensor_alternative(const Tensor &input, Variant &out) {
      if (!tensor_matches_alternative<Alternative>(input)) return false;

      using traits = tensor_t_alternative<Alternative>;
      using element_type = typename traits::element_type;
      using access_type = typename traits::access_type;

      if constexpr (mdspan_concepts::LayoutRight<Alternative>) {
        Tensor tensor = input.contiguous();
        out = make_right_tensor_t<element_type, traits::rank, access_type>(tensor);
      } else if constexpr (mdspan_concepts::LayoutStride<Alternative>) {
        Tensor tensor = input;
        out = make_tensor_t<element_type, traits::rank, access_type>(tensor);
      } else {
        static_assert(
          mdspan_concepts::LayoutRight<Alternative> || mdspan_concepts::LayoutStride<Alternative>,
          "Unsupported TensorT layout for make_tensor");
      }
      return true;
    }

    template <class Variant, std::size_t... Indices>
    Variant make_tensor_impl(const Tensor &tensor, std::index_sequence<Indices...>) {
      Variant out;
      const bool matched =
        (try_make_tensor_alternative<std::variant_alternative_t<Indices, Variant>>(tensor, out) ||
         ...);
      cytnx_error_msg(!matched,
                      "[ERROR] Tensor dtype/device/rank is not represented by this TensorT "
                      "variant.%s",
                      "\n");
      return out;
    }

  }  // namespace tensor_t_detail

  /**
   * @brief Create a typed TensorT variant from a legacy Tensor.
   *
   * `Variant` must be a `std::variant` whose alternatives are `TensorT` specializations. The
   * factory selects the alternative matching the Tensor rank, dtype, device backend, and requested
   * layout. Layout-right alternatives are made contiguous without mutating the input Tensor.
   */
  template <class Variant>
  Variant make_tensor(const Tensor &tensor) {
    return tensor_t_detail::make_tensor_impl<Variant>(
      tensor, std::make_index_sequence<std::variant_size_v<Variant>>{});
  }

}  // namespace cytnx

#endif  // CYTNX_TENSORT_TRAITS_HPP_
