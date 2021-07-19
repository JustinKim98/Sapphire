// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef  SAPPHIRE_COMPUTE_DENSE_ACTIVATION_BACKWARD_KERNEL_CUH
#define SAPPHIRE_COMPUTE_DENSE_ACTIVATION_BACKWARD_KERNEL_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__global__ void ReLUBackwardKernel(float* dx, const float* dy,
                                   const float* x, unsigned int totalSize);


__global__ void LeakyReLUBackwardKernel(float* dx, const float* dy,
                                        const float* x, float a,
                                        unsigned int totalSize);

__global__ void SoftMaxBackwardKernel(float* dx, const float* dy,
                                      const float* y, unsigned int totalSize,
                                      unsigned int unitSize);
}

#endif  // ! SAPPHIRE_COMPUTE_DENSE_ACTIVATION_BACKWARD_KERNEL_CUH
