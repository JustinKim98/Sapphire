// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_BASIC_BACKWARD_KERNEL_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_BASIC_BACKWARD_KERNEL_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__global__ void DotBackwardKernel(float* da, float* db, const float* dy,
                                  const float* a, const float* b,
                                  unsigned int offset, unsigned int launchSize,
                                  unsigned int totalSize,
                                  unsigned int inputStride,
                                  bool broadcastInputA, bool broadcastInputB);

__global__ void PowBackwardKernel(float* dx, const float* dy, const float* x,
                                  const float factor, unsigned totalSize);

__global__ void MeanBackwardKernel(float* dx,  const float* dy,
                                   unsigned int yTotalSize,
                                   unsigned int unitSize, unsigned int stride);
}

#endif
