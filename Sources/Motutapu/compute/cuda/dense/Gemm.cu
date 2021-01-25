// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/dense/GemmKernel.cuh>
#include <Motutapu/compute/cuda/dense/Gemm.cuh>


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
                         unsigned int paddedM,
                         unsigned int paddedN, unsigned int paddedK,
                         unsigned int batchSize, bool broadcastA,
                         bool broadcastB, bool broadcastC)
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
                streams[batchIdx]>>>(
                    ptrOut, ptrA, ptrB, ptrC, paddedK, paddedN,
                    chunkIdxK, chunkSize);
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

__host__ void GemmNormalFloat(float* out, float* A, float* B, float* C,
                              unsigned int paddedM, unsigned int paddedN,
                              unsigned int paddedK, unsigned int batchSize,
                              bool broadcastA, bool broadcastB, bool broadcastC)
{
    static constexpr unsigned int tileDim = 8;
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

            Gemm<<<numBlocks, chunkSize * chunkSize * 32,
                0, streams[batchIdx]>>>(
                    ptrOut, ptrA, ptrB, ptrC, paddedK, paddedN,
                    chunkIdxK);
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
}
