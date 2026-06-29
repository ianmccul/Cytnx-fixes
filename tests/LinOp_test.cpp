#include <gtest/gtest.h>
#include "cytnx.hpp"

using namespace cytnx;

namespace {

  class UniTensorIdentityOp : public LinOp {
   public:
    explicit UniTensorIdentityOp(const cytnx_uint64 nx)
        : LinOp("mv", nx, Type.Double, Device.cpu) {}

   protected:
    UniTensor matvec_impl(const UniTensor& Tin) override { return Tin; }
  };

  class UniTensorWrongOutputOp : public LinOp {
   public:
    UniTensorWrongOutputOp() : LinOp("mv", 4, Type.Double, Device.cpu) {}

   protected:
    UniTensor matvec_impl(const UniTensor&) override {
      return UniTensor::zeros({2, 3}, {}, Type.Double, Device.cpu).set_rowrank_(1);
    }
  };

}  // namespace

TEST(LinOp, UniTensorMatvecIgnoresConstructorNx) {
  UniTensorIdentityOp op(4);
  UniTensor input = UniTensor::zeros({2, 3}, {}, Type.Double, Device.cpu).set_rowrank_(1);
  EXPECT_NO_THROW({ op.matvec(input); });
}

TEST(LinOp, UniTensorMatvecRejectsOutputNxMismatch) {
  UniTensorWrongOutputOp op;
  UniTensor input = UniTensor::zeros({2, 2}, {}, Type.Double, Device.cpu).set_rowrank_(1);
  EXPECT_THROW({ op.matvec(input); }, std::exception);
}
