// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_CUDA_DENSE_GEMM_CUH
#define Sapphire_COMPUTE_CUDA_DENSE_GEMM_CUH

#include <cuda_fp16.h>
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void Gemm(unsigned int totalSize, float* out, float* A, float* B,
                   float* C, unsigned int M, unsigned int N, unsigned int K,
                   int deviceId);

__host__ void GemmMatrixWiseBroadcast(float* out, float* A, float* B, float* C,
                                      unsigned int M, unsigned int N,
                                      unsigned int K, unsigned int batchSize,
                                      bool broadcastA, bool broadcastB,
                                      bool broadcastC, int deviceId);

__host__ void GemmTensor(float* out, float* A, float* B, float* C,
                         unsigned int paddedM, unsigned int paddedN,
                         unsigned int paddedK, unsigned int batchSize,
                         bool broadcastA, bool broadcastB, bool broadcastC);

__host__ void GemmNormal(float* out, float* A, float* B, float* C,
                         unsigned int paddedM, unsigned int paddedN,
                         unsigned int paddedK, unsigned int batchSize,
                         bool broadcastA, bool broadcastB, bool broadcastC);

}  // namespace Sapphire::Compute::Cuda::Dense

#endif
