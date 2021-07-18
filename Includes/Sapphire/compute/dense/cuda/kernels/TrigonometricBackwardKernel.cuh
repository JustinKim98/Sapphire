// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_TRIGONOMETRIC_BACKWARD_KERNEL_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_TRIGONOMETRIC_BACKWARD_KERNEL_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__global__ void CosBackwardKernel(float* dx, const float* dy, const float* x,
                                  unsigned int totalSize);
__global__ void SinBackwardKernel(float* dx, const float* dy, const float* x,
                                  unsigned int totalSize);

__global__ void TanBackwardKernel(float* dx, const float* dy, const float* x,
                                  unsigned int totalSize);

__global__ void CoshBackwardKernel(float* dx, const float* dy, const float* x,
                                   unsigned int totalSize);

__global__ void SinhBackwardKernel(float* dx, const float* dy, const float* x,
                                   unsigned int totalSize);

__global__ void TanhBackwardKernel(float* dx, const float* dy, const float* x,
                                   unsigned int totalSize);

__global__ void ArcCosBackwardKernel(float* dx, const float* dy, const float* x,
                                     unsigned int totalSize);

__global__ void ArcSinBackwardKernel(float* dx, const float* dy, const float* x,
                                     unsigned int totalSize);

__global__ void ArcTanBackwardKernel(float* dx, const float* dy, const float* x,
                                     unsigned int totalSize);
__global__ void ArcCoshBackwardKernel(float* dx, const float* dy,
                                      const float* x, unsigned int totalSize);

__global__ void ArcSinhBackwardKernel(float* dx, const float* dy,
                                      const float* x, unsigned int totalSize);

__global__ void ArcTanhBackwardKernel(float* dx, const float* dy,
                                      const float* x, unsigned int totalSize);
}
#endif