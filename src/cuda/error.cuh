#ifndef ZFP_CUDA_ERROR_CUH
#define ZFP_CUDA_ERROR_CUH

#include <iostream>
#include <string>

namespace zfp {
namespace cuda {
namespace internal {

class ErrorCheck {
public:
  bool check(std::string msg)
  {
    error = cudaGetLastError();
    if (error != cudaSuccess) {
#ifdef ZFP_DEBUG
      std::cerr << "zfp::cuda : " << msg << " : " << cudaGetErrorString(error) << std::endl;
#endif
      return false;
    }
    return true;
  }

  cudaError error;
};

} // namespace internal
} // namespace cuda
} // namespace zfp

#endif
