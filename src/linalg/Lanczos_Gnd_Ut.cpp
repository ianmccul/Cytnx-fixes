#include "linalg.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace linalg {
    // Lanczos_Gnd_Ut is implemented together with Lanczos_Gnd in Lanczos_Gnd.cpp so the Tensor
    // and UniTensor entry points share the same residual-based Krylov core.
  }
}  // namespace cytnx

#endif
