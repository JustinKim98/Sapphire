// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CUDA_DENSE_INITIALIZE_KERNEL_CUH
#define MOTUTAPU_COMPUTE_CUDA_DENSE_INITIALIZE_KERNEL_CUH

#include <cuda_fp16.h>
#include <curand_kernel.h>

namespace Motutapu::Compute::Cuda::Dense
{
__global__ void NormalKernel(float* data, float mean, float sd,
                             unsigned int size, int seed);

__global__ void UniformKernel(float* data, float min, float max,
                              unsigned int size, int seed);

__global__ void ScalarKernel(float* data, float value, unsigned int size);
}  // namespace Motutapu::Compute::Cuda::Dense

#endif
