#ifndef CYTNX_MDSPAN_CONCEPTS_HPP_
#define CYTNX_MDSPAN_CONCEPTS_HPP_

#include "mdspan.hpp"

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace cytnx::mdspan_concepts {

  template <class Accessor>
  struct is_host_accessor : std::false_type {};

  template <class T>
  struct is_host_accessor<stdex::default_accessor<T>> : std::true_type {};

  template <class Accessor>
  concept HostAccessor = is_host_accessor<std::remove_cvref_t<Accessor>>::value;

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

}  // namespace cytnx::mdspan_concepts

#endif  // CYTNX_MDSPAN_CONCEPTS_HPP_
