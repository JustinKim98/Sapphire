// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CUDA_DENSE_DENSEMATRIX_CUH
#define MOTUTAPU_CUDA_DENSE_DENSEMATRIX_CUH

#include <Motutapu/compute/cuda/dense/Gemm.hpp>

namespace Motutapu::Cuda::Dense
{
__host__ void GemmTensorCore(half* out, half* A, half* B, unsigned int paddedN,
                             unsigned int paddedK, unsigned int paddedM,
                             unsigned int batchSize);

__host__ void GemmNormalFloat(float* out, float* A, float* B,
                              unsigned int paddedN, unsigned int paddedK,
                              unsigned int paddedM, unsigned int batchSize);

__host__ void GemmNormalHalf(half* out, half* A, half* B, unsigned int paddedN,
                             unsigned int paddedK, unsigned int paddedM,
                             unsigned int batchSize);
} // namespace Motutapu::Cuda::Dense

#endif
