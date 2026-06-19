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

  void LinOp::_print() {
    std::cout << "LinOp(type=" << this->_type << ", nx=" << this->_nx
              << ", dtype_hint=" << Type.getname(this->_dtype) << ", device=" << this->_device
              << ")" << std::endl;
  }

  Tensor LinOp::matvec(const Tensor &Tin) {
    cytnx_error_msg(tensor_numel(Tin) != this->_nx,
                    "[ERROR][LinOp] matvec input dimension %d does not match nx=%d\n",
                    tensor_numel(Tin), this->_nx);
    Tensor out = this->matvec_impl(Tin);
    cytnx_error_msg(tensor_numel(out) != this->_nx,
                    "[ERROR][LinOp] matvec output dimension %d does not match nx=%d\n",
                    tensor_numel(out), this->_nx);
    return out;
  }

  Tensor LinOp::matvec_impl(const Tensor &Tin) {
    cytnx_error_msg(
      true, "[ERROR][LinOp] LinOp with 'mv' type requires overriding matvec_impl before use.%s",
      "\n");
    return Tensor();
  }

  UniTensor LinOp::matvec(const UniTensor &Tin) {
    cytnx_error_msg(unitensor_numel(Tin) != this->_nx,
                    "[ERROR][LinOp] matvec input dimension %d does not match nx=%d\n",
                    unitensor_numel(Tin), this->_nx);
    UniTensor out = this->matvec_impl(Tin);
    cytnx_error_msg(unitensor_numel(out) != this->_nx,
                    "[ERROR][LinOp] matvec output dimension %d does not match nx=%d\n",
                    unitensor_numel(out), this->_nx);
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
