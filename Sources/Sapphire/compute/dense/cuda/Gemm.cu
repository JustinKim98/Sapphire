// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cublas_v2.h>
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/compute/dense/cuda/Basic.cuh>
#include <Sapphire/compute/dense/cuda/Gemm.cuh>
#include <Sapphire/compute/dense/cuda/GemmKernel.cuh>
#include <cassert>

namespace Sapphire::Compute::Cuda::Dense
{
//! All size parameters should be at least 1
//! batch sizes must be multiple of each other
__host__ void Gemm(unsigned int totalSize, float* out, float* A, float* B,
                   float* C, unsigned int M, unsigned int N, unsigned int K,
                   cublasHandle_t* handle)
{
    cublasSetMathMode(*handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    const auto strideA = M * K;
    const auto strideB = K * N;
    const auto strideOut = M * N;

    float* ptrA = A;
    float* ptrB = B;
    float* ptrC = C;
    float* ptrOut = out;

    CopyDeviceToDevice(ptrOut, ptrC, totalSize * sizeof(float));

    auto status = cublasGemmStridedBatchedEx(
        *handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, ptrB, CUDA_R_32F, N,
        strideB, ptrA, CUDA_R_32F, K, strideA, &beta, ptrOut, CUDA_R_32F, N,
        strideOut, totalSize / strideOut, CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    assert(status == CUBLAS_STATUS_SUCCESS);
}

//! Broadcasts operations matrix-wise
//! while broadcastC is false, broadcastOut must be false
__host__ void GemmMatrixWiseBroadcast(float* out, float* A, float* B, float* C,
                                      unsigned int M, unsigned int N,
                                      unsigned int K, unsigned int batchSize,
                                      bool broadcastA, bool broadcastB,
                                      bool broadcastC)
{
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);

    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    const auto strideA = (broadcastA ? 0 : (M * K));
    const auto strideB = (broadcastB ? 0 : (K * N));
    const auto strideOut = M * N;

    if (broadcastC)
    {
        CopyDeviceToDeviceBroadcast(out, C, M * N * batchSize * sizeof(float),
                                    M * N * sizeof(float));
    }
    else
        CopyDeviceToDevice(out, C, M * N * batchSize * sizeof(float));

    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                               &alpha, B, CUDA_R_32F, N, strideB, A, CUDA_R_32F,
                               K, strideA, &beta, out, CUDA_R_32F, N, strideOut,
                               batchSize, CUBLAS_COMPUTE_32F_FAST_TF32,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasDestroy(handle);
}

__host__ unsigned int Gcd(unsigned int a, unsigned int b)
{
    if (!a)
        return b;
    return Gcd(b % a, a);
}

__host__ unsigned int FindGCD(unsigned int arr[], int n)
{
    unsigned int result = arr[0];
    for (int i = 1; i < n; i++)
    {
        result = Gcd(arr[i], result);

        if (result == 1)
        {
            return 1;
        }
    }
    return result;
}


__host__ void GemmNormal(float* out, float* A, float* B, float* C,
                         unsigned int paddedM, unsigned int paddedN,
                         unsigned int paddedK, unsigned int batchSize,
                         bool broadcastA, bool broadcastB, bool broadcastC)
{
    auto* streams =
        static_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t) * batchSize));
    unsigned int blockSize = paddedM * paddedN / 1024 + 1;

    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        cudaStreamCreate(&streams[batchIdx]);

        float* ptrOut = out + paddedM * paddedN * batchIdx;
        const float* ptrA = A + paddedM * paddedK * (broadcastA ? 0 : batchIdx);
        const float* ptrB = B + paddedK * paddedN * (broadcastB ? 0 : batchIdx);
        const float* ptrC = C + paddedM * paddedN * (broadcastC ? 0 : batchIdx);
        GemmSimple<<<blockSize, 1024, 0, streams[batchIdx]>>>(
            ptrOut, ptrA, ptrB, ptrC, paddedM, paddedN, paddedK);
    }

    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        cudaStreamSynchronize(streams[batchIdx]);
        cudaStreamDestroy(streams[batchIdx]);
    }

    free(streams);
}
}  // namespace Sapphire::Compute::Cuda::Dense
