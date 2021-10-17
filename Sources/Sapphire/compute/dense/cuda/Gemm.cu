// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cublas_v2.h>
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/compute/dense/cuda/Gemm.cuh>
#include <Sapphire/compute/dense/cuda/kernels/GemmKernel.cuh>
#include <cassert>

namespace Sapphire::Compute::Dense::Cuda
{
//! All size parameters should be at least 1
//! batch sizes must be multiple of each other
__host__ void Gemm(unsigned int totalSize, float* out, const float* A,
                   const float* B, unsigned int M, unsigned int N,
                   unsigned int K,
                   int deviceId)
{
    const auto tid = std::this_thread::get_id();
    if (!Util::ResourceManager::HasCublasHandle(deviceId, tid))
    {
        Util::ResourceManager::AddCublasHandle(deviceId, tid);
    }
    auto* handle = Util::ResourceManager::GetCublasHandle(
        deviceId, tid);
    cublasSetMathMode(*handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    const auto strideA = M * K;
    const auto strideB = K * N;
    const auto strideOut = M * N;

    const float* ptrA = A;
    const float* ptrB = B;
    float* ptrOut = out;

    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        *handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N),
        static_cast<int>(M), static_cast<int>(K), &alpha, ptrB, CUDA_R_32F,
        static_cast<int>(N), strideB, ptrA, CUDA_R_32F, static_cast<int>(K),
        strideA, &beta, ptrOut, CUDA_R_32F, static_cast<int>(N), strideOut,
        static_cast<int>(totalSize / strideOut), CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP))
}

//! Broadcasts operations matrix-wise
//! while broadcastC is false, broadcastOut must be false
__host__ void GemmMatrixWiseBroadcast(float* out, const float* A,
                                      const float* B, 
                                      unsigned int M, unsigned int N,
                                      unsigned int K, unsigned int batchSize,
                                      bool broadcastA, bool broadcastB,int deviceId)
{
    const auto tid = std::this_thread::get_id();
    if (!Util::ResourceManager::HasCublasHandle(deviceId, tid))
    {
        Util::ResourceManager::AddCublasHandle(deviceId, tid);
    }
    auto* handle = Util::ResourceManager::GetCublasHandle(deviceId, tid);

    cublasSetMathMode(*handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    const auto strideA = (broadcastA ? 0 : (M * K));
    const auto strideB = (broadcastB ? 0 : (K * N));
    const auto strideOut = M * N;


    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        *handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N),
        static_cast<int>(M), static_cast<int>(K), &alpha, B, CUDA_R_32F,
        static_cast<int>(N), strideB, A, CUDA_R_32F, static_cast<int>(K),
        strideA, &beta, out, CUDA_R_32F, static_cast<int>(N), strideOut,
        static_cast<int>(batchSize), CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP))
}
} // namespace Sapphire::Compute::Dense::Cuda
