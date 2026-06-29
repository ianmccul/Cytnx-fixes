#include "LinOp.hpp"
#include "Tensor.hpp"
#include "utils/vec_print.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {

  namespace {

    cytnx_uint64 tensor_numel(const Tensor &Tin) {
      cytnx_uint64 numel = 1;
      for (const auto dim : Tin.shape()) numel *= dim;
      return numel;
    }

    cytnx_uint64 unitensor_numel(const UniTensor &Tin) {
      if (Tin.uten_type() == UTenType.Dense) return tensor_numel(Tin.get_block_());
      if (Tin.uten_type() == UTenType.Block || Tin.uten_type() == UTenType.BlockFermionic) {
        cytnx_uint64 numel = 0;
        for (const auto &block : Tin.get_blocks_()) numel += tensor_numel(block);
        return numel;
      }
      return 0;
    }

  }  // namespace

  void LinOp::_print() { std::cout << "LinOp(type=" << this->_type << ")" << std::endl; }

  UniTensor LinOp::matvec(const UniTensor &Tin) {
    const auto input_numel = unitensor_numel(Tin);
    UniTensor out = this->matvec_impl(Tin);
    cytnx_error_msg(unitensor_numel(out) != input_numel,
                    "[ERROR][LinOp] matvec output dimension %d does not match input dimension %d\n",
                    unitensor_numel(out), input_numel);
    return out;
  }

  UniTensor LinOp::matvec_impl(const UniTensor &Tin) {
    cytnx_error_msg(
      true, "[ERROR][LinOp] LinOp with 'mv' type requires overriding matvec_impl before use.%s",
      "\n");
    return UniTensor();
  }

}  // namespace cytnx
#endif
