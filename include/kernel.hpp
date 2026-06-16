#ifndef CYTNX_KERNEL_HPP_
#define CYTNX_KERNEL_HPP_

#include "cytnx_error.hpp"

#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace cytnx {

  template <class T>
  struct is_variant : std::false_type {};

  template <class... Alternatives>
  struct is_variant<std::variant<Alternatives...>> : std::true_type {};

  /// True if `T` is a `std::variant`.
  template <class T>
  concept Variant = is_variant<std::remove_cvref_t<T>>::value;

  namespace kernel_detail {

    template <class T>
    struct dispatch_alternatives {
      using type = std::tuple<T>;
    };

    template <class T, class From>
    struct copy_cv {
      using from_type = std::remove_reference_t<From>;
      using const_type = std::conditional_t<std::is_const_v<from_type>, std::add_const_t<T>, T>;
      using type = std::conditional_t<std::is_volatile_v<from_type>,
                                      std::add_volatile_t<const_type>, const_type>;
    };

    template <class T, class From>
    using copy_cv_t = typename copy_cv<T, From>::type;

    template <class Alternative, class VariantArg>
    struct variant_dispatch_alternative {
      using cv_alternative = copy_cv_t<Alternative, VariantArg>;
      using type = std::conditional_t<std::is_lvalue_reference_v<VariantArg>, cv_alternative &,
                                      cv_alternative>;
    };

    template <class Alternative, class VariantArg>
    using variant_dispatch_alternative_t =
      typename variant_dispatch_alternative<Alternative, VariantArg>::type;

    template <class VariantArg, class Variant>
    struct variant_dispatch_alternatives;

    template <class VariantArg, class... Alternatives>
    struct variant_dispatch_alternatives<VariantArg, std::variant<Alternatives...>> {
      using type = std::tuple<variant_dispatch_alternative_t<Alternatives, VariantArg>...>;
    };

    template <class T>
      requires Variant<T>
    struct dispatch_alternatives<T> {
      using type = typename variant_dispatch_alternatives<T, std::remove_cvref_t<T>>::type;
    };

    template <class T>
    using dispatch_alternatives_t = typename dispatch_alternatives<T>::type;

    template <class F, class Accumulated, class... AlternativeTuples>
    struct any_dispatch_invocable;

    template <class F, class... Args>
    concept kernel_invocable = requires(F &&f, Args &&...args) {
      run_kernel(std::forward<F>(f), std::forward<Args>(args)...);
    };

    template <class F, class... Accumulated>
    struct any_dispatch_invocable<F, std::tuple<Accumulated...>> {
      static constexpr bool value = kernel_invocable<F, Accumulated...>;
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

  }  // namespace kernel_detail

  /**
   * @brief True if at least one Cartesian product of dispatch alternatives is invocable by `F`.
   *
   * Each argument type may be a concrete type or a `std::variant`; non-variants are treated as
   * single-alternative dispatch arguments.
   */
  template <class F, class... Args>
  concept AnyDispatchInvocable =
    kernel_detail::any_dispatch_invocable<F, std::tuple<>,
                                          kernel_detail::dispatch_alternatives_t<Args>...>::value;

  /**
   * @brief Invoke a kernel over concrete or variant arguments.
   *
   * Concrete arguments are forwarded directly. Variant arguments are visited, and the active
   * alternative combination must be invocable by `kernel`; otherwise `error_message` is reported.
   */
  template <class Kernel, class... Args>
    requires AnyDispatchInvocable<Kernel, Args...>
  decltype(auto) invoke_kernel(Kernel &&kernel, const char *error_message, Args &&...args) {
    return kernel_detail::dispatch_visit(
      [&kernel, error_message](auto &&...active) -> decltype(auto) {
        if constexpr (kernel_detail::kernel_invocable<Kernel, decltype(active)...>) {
          return run_kernel(std::forward<Kernel>(kernel),
                            std::forward<decltype(active)>(active)...);
        } else {
          cytnx_error_msg(true, "%s%s", error_message, "\n");
        }
      },
      std::forward<Args>(args)...);
  }

}  // namespace cytnx

#endif  // CYTNX_KERNEL_HPP_
