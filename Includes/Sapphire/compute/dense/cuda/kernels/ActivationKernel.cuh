// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_ACTAVATION_KERNEL_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_ACTAVATION_KERNEL_CUH\

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
//! Total size must be multiple of unitSize
__global__ void SoftMaxKernel(float* y, const float* x, unsigned int totalSize,
                              unsigned int unitSize);

__global__ void LeakyReLUKernel(float* y, const float* x,
                                float a, unsigned int totalSize);

__global__ void ReLUKernel(float* y, const float* x, unsigned int totalSize);
}

#endif