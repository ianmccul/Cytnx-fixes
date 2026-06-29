#ifndef CYTNX_LINOP_H_
#define CYTNX_LINOP_H_

#include "Device.hpp"
#include "Type.hpp"
#include "cytnx_error.hpp"
#include <vector>
#include <fstream>
#include <functional>
#include "intrusive_ptr_base.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
namespace cytnx {

  class LinOp {
   private:
    // type, retained for construction compatibility. Only "mv" is supported.
    std::string _type;

   public:
    /// @cond
    // we need driver of void f(nx,vin,vout)
    /// @endcond

    /**
    @brief Linear Operator class for iterative solvers.
    @param type the type of operator, currently it can only be "mv" (matvec).
    @param nx retained for source compatibility; iterative solvers derive the vector dimension from
    the input vector.
    @param dtype retained for source compatibility; iterative solvers derive the working dtype from
    the input vector.
    @param device retained for source compatibility; iterative solvers derive the device from the
    input vector.

    ## Details:
        The LinOp class is a class that defines a custom Linear operation acting on a
    UniTensor. To use, inherit this class and override the matvec_impl function. See the following
    examples for how to use them.

    ## Example:
    ### python API:
    \include example/LinOp/init.py
    #### output>
    \verbinclude example/LinOp/init.py.out

    */
    LinOp(const std::string &type, const cytnx_uint64 &nx, const unsigned int &dtype,
          const int &device = Device.cpu) {
      cytnx_error_msg(type != "mv",
                      "[ERROR][LinOp] currently only type=\"mv\" (matvec) can be used.%s", "\n");
      this->_type = type;
      cytnx_error_msg(device < -1 || device >= Device.Ngpus, "[ERROR] invalid device.%s", "\n");
    };

    unsigned int dtype() const { return Type.Void; }

    int device() const { return Device.cpu; };
    cytnx_uint64 nx() const { return 0; };

    void _print();

    UniTensor matvec(const UniTensor &Tin);

   protected:
    virtual UniTensor matvec_impl(const UniTensor &Tin);
  };

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_LINOP_H_
