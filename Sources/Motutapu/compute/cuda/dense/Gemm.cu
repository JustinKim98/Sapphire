// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cublas_v2.h>
#include <Motutapu/compute/cuda/CudaParams.cuh>
#include <Motutapu/compute/cuda/dense/Gemm.cuh>
#include <Motutapu/compute/cuda/dense/GemmKernel.cuh>
#include <cassert>

namespace Motutapu::Compute::Cuda::Dense
{
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

__host__ void GemmTensor(float* out, float* A, float* B, float* C,
                         unsigned int paddedM, unsigned int paddedN,
                         unsigned int paddedK, unsigned int batchSize,
                         bool broadcastA, bool broadcastB, bool broadcastC)
{
    static constexpr unsigned int tileDim = 16;
    const auto chunkDimM = paddedM / 16;
    const auto chunkDimK = paddedK / 16;
    const auto chunkDimN = paddedN / 16;

    unsigned int arr[] = { chunkDimM, chunkDimK, chunkDimN };

    const auto maxTileSize = FindGCD(arr, sizeof(arr) / sizeof(arr[0]));

    unsigned int chunkSize;
    if (maxTileSize % 2 == 1)
    {
        chunkSize = 1;
    }
    else
    {
        if (maxTileSize % 4 == 0)
            chunkSize = 4;
        else
            chunkSize = 2;
    }

    auto* streams =
        static_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t) * batchSize));

    for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        cudaStreamCreate(&streams[batchIdx]);
    }
    for (unsigned int chunkIdxK = 0; chunkIdxK * chunkSize * tileDim < paddedN;
         ++chunkIdxK)
    {
        for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            float* ptrOut = out + paddedM * paddedN * batchIdx;
            const float* ptrA =
                A + paddedM * paddedK * (broadcastA ? 1 : batchIdx);
            const float* ptrB =
                B + paddedK * paddedN * (broadcastB ? 1 : batchIdx);
            const float* ptrC =
                C + paddedM * paddedN * (broadcastC ? 1 : batchIdx);

            const dim3 numBlocks(chunkDimM, chunkDimN);

            WmmaGemm<<<numBlocks, chunkSize * chunkSize * 32, 0,
                       streams[batchIdx]>>>(ptrOut, ptrA, ptrB, ptrC, paddedK,
                                            paddedN, chunkIdxK, chunkSize);
        }

        for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            cudaStreamSynchronize(streams[batchIdx]);
        }
    }

    for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        cudaStreamDestroy(streams[batchIdx]);
    }

    free(streams);
}

//! M, N, K should be multiple of 8
//!
__host__ void GemmCublas(float* out, float* A, float* B, float* C,
                         unsigned int paddedM, unsigned int paddedN,
                         unsigned int paddedK, unsigned int batchSize,
                         bool broadcastA, bool broadcastB, bool broadcastC)
{
    assert(intptr_t(A) % 16 == 0);
    assert(intptr_t(B) % 16 == 0);
    assert(intptr_t(out) % 16 == 0);
    assert(intptr_t(A + paddedK) % 16 == 0);
    assert(intptr_t(B + paddedN) % 16 == 0);
    assert(intptr_t(out + paddedN) % 16 == 0);
    assert(paddedM % 8 == 0);
    assert(paddedN % 8 == 0);
    assert(paddedK % 8 == 0);

    auto* streams =
        static_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t) * batchSize));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);

    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    const auto strideA = (broadcastA ? 0 : (paddedM * paddedK));
    const auto strideB = (broadcastB ? 0 : (paddedK * paddedN));
    const auto strideC = (broadcastC ? 0 : (paddedM * paddedN));

    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        cudaStreamCreate(&streams[batchIdx]);
        float* ptrOut = out + paddedM * paddedN * batchIdx;
        const float* ptrA = A + batchIdx * strideA;
        const float* ptrB = B + batchIdx * strideB;
        const float* ptrC = C + batchIdx * strideC;
        cudaMemcpyAsync(ptrOut, ptrC, paddedM * paddedN * sizeof(float),
                        cudaMemcpyDeviceToDevice, streams[batchIdx]);
        cublasSetStream(handle, streams[batchIdx]);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, paddedN, paddedM, paddedK,
                    &alpha, ptrB, paddedN, ptrA, paddedK, &beta, ptrOut,
                    paddedN);
    }

    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        cudaStreamSynchronize(streams[batchIdx]);
        cudaStreamDestroy(streams[batchIdx]);
    }

    free(streams);
    cublasDestroy(handle);
}

__host__ void GemmCublas(float* out, float* A, float* B, unsigned int paddedM,
                         unsigned int paddedN, unsigned int paddedK,
                         unsigned int batchSize, bool broadcastA,
                         bool broadcastB)
{
    auto* streams =
        static_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t) * batchSize));
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    const auto strideA = (broadcastA ? 0 : (paddedM * paddedK));
    const auto strideB = (broadcastB ? 0 : (paddedK * paddedN));

    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        cudaStreamCreate(&streams[batchIdx]);
        float* ptrOut = out + paddedM * paddedN * batchIdx;
        const float* ptrA = A + batchIdx * strideA;
        const float* ptrB = B + batchIdx * strideB;
        cublasSetStream(handle, streams[batchIdx]);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, paddedN, paddedM, paddedK,
                    &alpha, ptrB, paddedN, ptrA, paddedK, &beta, ptrOut,
                    paddedN);
    }

    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        cudaStreamSynchronize(streams[batchIdx]);
        cudaStreamDestroy(streams[batchIdx]);
    }

    free(streams);
    cublasDestroy(handle);
}

__host__ void GemmNormal(float* out, float* A, float* B, float* C,
                         unsigned int paddedM, unsigned int paddedN,
                         unsigned int paddedK, unsigned int batchSize,
                         bool broadcastA, bool broadcastB, bool broadcastC)
{
    // GemmSimple<<<1, 1024>>>(out, A, B, C, paddedM, paddedN, paddedK);

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

    //    static constexpr unsigned int tileDim = 8;
    //    const auto chunkDimM = paddedM / tileDim;
    //    const auto chunkDimK = paddedK / tileDim;
    //    const auto chunkDimN = paddedN / tileDim;
    //
    //    unsigned int arr[] = { chunkDimM, chunkDimK, chunkDimN };
    //
    //    const auto maxTileSize = FindGCD(arr, sizeof(arr) / sizeof(arr[0]));
    //
    //    unsigned int chunkSize;
    //    if (maxTileSize % 2 == 1)
    //    {
    //        chunkSize = 1;
    //    }
    //    else
    //    {
    //        if (maxTileSize % 4 == 0)
    //            chunkSize = 4;
    //        else
    //            chunkSize = 2;
    //    }
    //
    //    for (unsigned int chunkIdxK = 0; chunkIdxK * chunkSize * tileDim <
    //    paddedK;
    //         ++chunkIdxK)
    //    {
    //        for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    //        {
    //            float* ptrOut = out + paddedM * paddedN * batchIdx;
    //            const float* ptrA =
    //                A + paddedM * paddedK * (broadcastA ? 0 : batchIdx);
    //            const float* ptrB =
    //                B + paddedK * paddedN * (broadcastB ? 0 : batchIdx);
    //            const float* ptrC =
    //                C + paddedM * paddedN * (broadcastC ? 0 : batchIdx);
    //
    //            const unsigned int numBlocks = chunkDimM * chunkDimN;
    //            const unsigned int numThreads =
    //                chunkSize * tileDim * chunkSize * tileDim;
    //            const unsigned int sharedMemSize =
    //                (tileDim * chunkSize) * (tileDim * chunkSize + 1) * 2;
    //
    //            Gemm<<<numBlocks, numThreads, sharedMemSize *
    //            sizeof(float)>>>(
    //                ptrOut, ptrA, ptrB, ptrC, paddedM, paddedN, paddedK,
    //                chunkIdxK, chunkSize, chunkDimN);
    //        }
    //    }
}
}  // namespace Motutapu::Compute::Cuda::Dense
