// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CUDA_DENSE_INITIALIZE_KERNEL_CUH
#define MOTUTAPU_COMPUTE_CUDA_DENSE_INITIALIZE_KERNEL_CUH

#include <curand_kernel.h>
#include <cuda_fp16.h>

namespace Motutapu::Compute::Cuda::Dense
{
__global__ void initRandomKernel(curandState* state);

__global__ void NormalFloatKernel(float* data, float mean, float sd,
                                  unsigned int size,
                                  curandState* state);

__global__ void NormalHalfKernel(half* data, half mean, half sd,
                                 unsigned int size,
                                 curandState* state);

__global__ void ScalarFloatKernel(float* data,
                                  float value, unsigned int size);

__global__ void ScalarHalfKernel(half* data,
                                 half value, unsigned int size);
}

#endif
