#include <vector>
#include <map>
#include <random>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>

#include "cytnx.hpp"
// #include "../include/cytnx_error.hpp"
#include "complex.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cytnx;

#ifdef BACKEND_TORCH
#else

class PyLinOp : public LinOp {
 public:
  /* inherit constructor */
  using LinOp::LinOp;

  UniTensor matvec_impl(const UniTensor &Tin) override {
    PYBIND11_OVERLOAD_NAME(UniTensor, /* Return type */
                           LinOp, /* Parent class */
                           "matvec", /* Name of function in Python */
                           matvec_impl, /* Name of function in C++ */
                           Tin /* Argument(s) */
    );
  }
};

void linop_binding(py::module &m) {
  py::class_<LinOp, PyLinOp>(m, "LinOp")
    .def(py::init([](const std::string &type, const cytnx_uint64 &nx, py::object dtype,
                     const int &device) {
           cytnx_error_msg(dtype.is_none(), "[ERROR][LinOp] dtype must be supplied.%s", "\n");
           if (PyErr_WarnEx(PyExc_UserWarning,
                            "[WARNING][LinOp] constructor nx, dtype, and device metadata are "
                            "ignored. Krylov solvers derive dimension, dtype, and device from "
                            "the input vector.",
                            1) < 0) {
             throw py::error_already_set();
           }
           unsigned int dtype_hint = dtype.cast<unsigned int>();
           return new PyLinOp(type, nx, dtype_hint, device);
         }),
         py::arg("type"), py::arg("nx"), py::arg("dtype"), py::arg("device") = (int)Device.cpu)
    .def(
      "set_dtype",
      [](LinOp &, const unsigned int &) {
        throw std::runtime_error(
          "[ERROR][LinOp] set_dtype() is no longer supported. LinOp dtype metadata is ignored; "
          "Krylov solvers derive the working dtype from the input vector.");
      },
      py::arg("dtype"))
    .def("dtype", &LinOp::dtype)
    .def(
      "matvec", [](LinOp &self, const UniTensor &Tin) -> UniTensor { return self.matvec(Tin); },
      py::arg("Tin"))
    .def(
      "set_device",
      [](LinOp &, const int &) {
        throw std::runtime_error(
          "[ERROR][LinOp] set_device() is no longer supported. LinOp device metadata is ignored; "
          "Krylov solvers derive the device from the input vector.");
      },
      py::arg("device"))
    .def("device", &LinOp::device)
    .def("nx", &LinOp::nx)
    .def(
      "__repr__",
      [](cytnx::LinOp &self) -> std::string {
        self._print();
        return std::string("");
      },
      py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

    ;  // end of object
}

#endif
