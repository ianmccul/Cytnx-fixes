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

    // nx
    cytnx_uint64 _nx;

    // Operator coefficient dtype hint. Type.Void can be supplied explicitly for unknown.
    unsigned int _dtype;

    // device
    int _device;

   public:
    /// @cond
    // we need driver of void f(nx,vin,vout)
    /// @endcond

    /**
    @brief Linear Operator class for iterative solvers.
    @param type the type of operator, currently it can only be "mv" (matvec).
    @param nx the dimension of the vector space on which this linear operator acts.
    @param dtype operator coefficient dtype hint. Type.Void can be supplied explicitly for unknown.
    @param device the Operator's on device.

    ## Note:
        1. LinOp dtype is a promotion hint, not an input/output contract. Concrete operators may
    promote dtype internally, and iterative algorithms choose their working dtype from the input
    vector and this hint.

    ## Details:
        The LinOp class is a class that defines a custom Linear operation acting on a Tensor or
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
      this->_nx = nx;
      this->_dtype = dtype;
      cytnx_error_msg(device < -1 || device >= Device.Ngpus, "[ERROR] invalid device.%s", "\n");
      this->_device = device;
    };

    void set_dtype(const unsigned int &dtype) { this->_dtype = dtype; }
    unsigned int dtype() const { return this->_dtype; }

    void set_device(const int &device) {
      cytnx_error_msg(device < -1 || device >= Device.Ngpus, "[ERROR] invalid device.%s", "\n");
      this->_device = device;
    };
    int device() const { return this->_device; };
    cytnx_uint64 nx() const { return this->_nx; };

    void _print();

    Tensor matvec(const Tensor &Tin);

    UniTensor matvec(const UniTensor &Tin);

   protected:
    virtual Tensor matvec_impl(const Tensor &Tin);
    virtual UniTensor matvec_impl(const UniTensor &Tin);
  };

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_LINOP_H_
