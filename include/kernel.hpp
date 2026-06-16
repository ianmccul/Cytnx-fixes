#ifndef CYTNX_KERNEL_HPP_
#define CYTNX_KERNEL_HPP_

#include "cytnx_error.hpp"
#include "mdspan_concepts.hpp"

#include <concepts>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>
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

    template <class T>
    std::string type_name() {
      return typeid(std::remove_cvref_t<T>).name();
    }

    template <class Kernel>
    concept has_kernel_name = requires { std::remove_cvref_t<Kernel>::name; };

    template <class Kernel>
      requires has_kernel_name<Kernel>
    constexpr std::string_view kernel_name() {
      return std::remove_cvref_t<Kernel>::name;
    }

    template <class Kernel>
      requires(!has_kernel_name<Kernel>)
    std::string kernel_name() {
      return type_name<Kernel>();
    }

    template <class T>
    concept has_device = requires(const T &value) {
      { value.device() } -> std::convertible_to<int>;
    };

    template <class T>
    concept tensor_metadata_like = requires(const T &value) {
      { value.rank() } -> std::convertible_to<std::size_t>;
      { value.shape() };
      { value.dtype_str() } -> std::convertible_to<std::string>;
      { value.device_str() } -> std::convertible_to<std::string>;
      { value.is_contiguous() } -> std::convertible_to<bool>;
    };

    template <class Sequence>
    void append_sequence(std::ostringstream &out, const Sequence &sequence) {
      out << '[';
      for (std::size_t i = 0; i < sequence.size(); ++i) {
        if (i != 0) out << ", ";
        out << sequence[i];
      }
      out << ']';
    }

    template <mdspan_concepts::MdspanView View>
    void append_extents(std::ostringstream &out, const View &view) {
      out << '[';
      for (std::size_t axis = 0; axis < std::remove_cvref_t<View>::rank(); ++axis) {
        if (axis != 0) out << ", ";
        out << view.extent(axis);
      }
      out << ']';
    }

    template <mdspan_concepts::MdspanView View>
    void append_strides(std::ostringstream &out, const View &view) {
      out << '[';
      for (std::size_t axis = 0; axis < std::remove_cvref_t<View>::rank(); ++axis) {
        if (axis != 0) out << ", ";
        out << view.stride(axis);
      }
      out << ']';
    }

    template <class Layout>
    std::string_view layout_name() {
      if constexpr (std::is_same_v<std::remove_cvref_t<Layout>, stdex::layout_right>) {
        return "layout_right";
      } else if constexpr (std::is_same_v<std::remove_cvref_t<Layout>, stdex::layout_stride>) {
        return "layout_stride";
      } else {
        return {};
      }
    }

    template <class Accessor>
    std::string_view accessor_name() {
      if constexpr (mdspan_concepts::HostAccessor<Accessor>) {
        return "host";
      } else if constexpr (mdspan_concepts::CudaAccessor<Accessor>) {
        return "cuda";
      } else {
        return {};
      }
    }

    template <mdspan_concepts::MdspanView Arg>
    void append_mdspan_argument(std::ostringstream &out, const Arg &arg) {
      using arg_type = std::remove_cvref_t<Arg>;
      out << "    rank: " << arg_type::rank() << '\n';
      out << "    extents: ";
      append_extents(out, arg);
      out << '\n';
      out << "    strides: ";
      append_strides(out, arg);
      out << '\n';
      out << "    element_type: " << type_name<typename arg_type::element_type>() << '\n';
      const auto layout = layout_name<typename arg_type::layout_type>();
      out << "    layout: "
          << (layout.empty() ? type_name<typename arg_type::layout_type>() : std::string(layout))
          << '\n';
      const auto accessor = accessor_name<typename arg_type::accessor_type>();
      out << "    access: "
          << (accessor.empty() ? type_name<typename arg_type::accessor_type>()
                               : std::string(accessor))
          << '\n';
      if constexpr (has_device<Arg>) {
        out << "    device: " << arg.device() << '\n';
      }
    }

    template <tensor_metadata_like Arg>
    void append_tensor_metadata_argument(std::ostringstream &out, const Arg &arg) {
      out << "    rank: " << arg.rank() << '\n';
      out << "    shape: ";
      append_sequence(out, arg.shape());
      out << '\n';
      out << "    dtype: " << arg.dtype_str() << '\n';
      out << "    device: " << arg.device_str() << '\n';
      out << "    contiguous: " << (arg.is_contiguous() ? "true" : "false") << '\n';
    }

    template <class Arg>
    void append_argument(std::ostringstream &out, std::size_t index, const Arg &arg) {
      out << "  arg" << index << ":\n";
      out << "    type: " << type_name<Arg>() << '\n';
      if constexpr (mdspan_concepts::MdspanView<std::remove_cvref_t<Arg>>) {
        append_mdspan_argument(out, arg);
      } else if constexpr (tensor_metadata_like<std::remove_cvref_t<Arg>>) {
        append_tensor_metadata_argument(out, arg);
      }
    }

    template <class Kernel, class... Args>
    std::string generic_kernel_error(Args &&...args) {
      std::ostringstream out;
      out << "[ERROR] No matching backend kernel found for " << kernel_name<Kernel>() << ".\n";
      out << "arguments:\n";
      std::size_t index = 0;
      (append_argument(out, index++, args), ...);
      return out.str();
    }

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
   * @brief Generic failure diagnostic for an active kernel dispatch combination.
   *
   * Backend kernels can overload this function beside their `run_kernel` overloads to provide
   * operation-specific diagnostics. This fallback is used when no better overload is available.
   */
  template <class Kernel, class... Args>
  std::string describe_kernel_error(Kernel &&, Args &&...args) {
    return kernel_detail::generic_kernel_error<Kernel, Args...>(std::forward<Args>(args)...);
  }

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
   * alternative combination must be invocable by `kernel`; otherwise a failure diagnostic is
   * reported.
   */
  template <class Kernel, class... Args>
    requires AnyDispatchInvocable<Kernel, Args...>
  decltype(auto) invoke_kernel(Kernel &&kernel, Args &&...args) {
    return kernel_detail::dispatch_visit(
      [&kernel](auto &&...active) -> decltype(auto) {
        if constexpr (kernel_detail::kernel_invocable<Kernel, decltype(active)...>) {
          return run_kernel(std::forward<Kernel>(kernel),
                            std::forward<decltype(active)>(active)...);
        } else {
          const std::string message = describe_kernel_error(
            std::forward<Kernel>(kernel), std::forward<decltype(active)>(active)...);
          cytnx_error_msg(true, "%s", message.c_str());
        }
      },
      std::forward<Args>(args)...);
  }

}  // namespace cytnx

#endif  // CYTNX_KERNEL_HPP_
