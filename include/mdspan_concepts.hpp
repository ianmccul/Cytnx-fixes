#ifndef CYTNX_MDSPAN_CONCEPTS_HPP_
#define CYTNX_MDSPAN_CONCEPTS_HPP_

#include "mdspan.hpp"

#include <concepts>
#include <complex>
#include <cstddef>
#include <type_traits>

namespace cytnx::mdspan_concepts {

  namespace detail {

    template <class T>
    struct real_element {
      using type = std::remove_cv_t<T>;
    };

    template <class T>
    struct real_element<std::complex<T>> {
      using type = T;
    };

  }  // namespace detail

  template <class T>
  using real_element_t = typename detail::real_element<std::remove_cv_t<T>>::type;

  template <class Accessor>
  struct is_host_accessor : std::false_type {};

  template <class T>
  struct is_host_accessor<stdex::default_accessor<T>> : std::true_type {};

  template <class Accessor>
  struct is_cuda_accessor : std::false_type {};

  template <class T>
  struct is_cuda_accessor<stdex::cuda_accessor<T>> : std::true_type {};

  template <class Accessor>
  concept HostAccessor = is_host_accessor<std::remove_cvref_t<Accessor>>::value;

  template <class Accessor>
  concept CudaAccessor = is_cuda_accessor<std::remove_cvref_t<Accessor>>::value;

  template <class View>
  concept MdspanView = requires(View view, std::size_t axis) {
    typename View::element_type;
    typename View::layout_type;
    typename View::accessor_type;
    typename View::data_handle_type;
    { View::rank() } -> std::convertible_to<std::size_t>;
    { view.extent(axis) } -> std::convertible_to<std::size_t>;
    { view.stride(axis) } -> std::convertible_to<std::size_t>;
    { view.data_handle() } -> std::convertible_to<typename View::data_handle_type>;
    { view.accessor() } -> std::same_as<const typename View::accessor_type &>;
  };

  template <class View>
  concept HostAccessible = MdspanView<View> && HostAccessor<typename View::accessor_type>;

  template <class View>
  concept CudaAccessible = MdspanView<View> && CudaAccessor<typename View::accessor_type>;

  template <class View, std::size_t Rank>
  concept RankOf = MdspanView<View> && View::rank() == Rank;

  template <class View>
  concept Vector = RankOf<View, 1>;

  template <class View>
  concept Matrix = RankOf<View, 2>;

  template <class View>
  concept LayoutRight =
    MdspanView<View> && std::is_same_v<typename View::layout_type, stdex::layout_right>;

  template <class View>
  concept LayoutStride =
    MdspanView<View> && std::is_same_v<typename View::layout_type, stdex::layout_stride>;

  template <class View>
  concept LayoutRightVector = Vector<View> && LayoutRight<View>;

  template <class View>
  concept LayoutRightMatrix = Matrix<View> && LayoutRight<View>;

  template <class First, class... Rest>
  concept SameElementType =
    (... && std::same_as<typename First::element_type, typename Rest::element_type>);

  template <class View>
  struct RealElementOf {
    using element_type = real_element_t<typename View::element_type>;
  };

}  // namespace cytnx::mdspan_concepts

#endif  // CYTNX_MDSPAN_CONCEPTS_HPP_
