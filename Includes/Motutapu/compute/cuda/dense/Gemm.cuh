// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CUDA_DENSE_GEMM_CUH
#define MOTUTAPU_COMPUTE_CUDA_DENSE_GEMM_CUH

#include <Motutapu/compute/cuda/CudaParams.hpp>
#include <cuda_fp16.h>

namespace Motutapu::Compute::Cuda::Dense
{
__host__ void GemmTensor(float* out, float* A, float* B, float* C,
                         unsigned int paddedM, unsigned int paddedN,
                         unsigned int paddedK, unsigned int batchSize,
                         bool broadcastA, bool broadcastB, bool broadcastC);

__host__ void GemmNormalFloat(float* out, float* A, float* B, float* C,
                              unsigned int paddedM, unsigned int paddedN,
                              unsigned int paddedK, unsigned int batchSize,
                              bool broadcastA, bool broadcastB,
                              bool broadcastC);

} // namespace Motutapu::Cuda::Dense

#endif
