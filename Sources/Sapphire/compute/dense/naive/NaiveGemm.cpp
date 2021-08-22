// Copyright (c) 2020, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/naive/NaiveGemm.hpp>
#include <cstdlib>

namespace Sapphire::Compute::Dense::Naive
{
void NaiveGemm(unsigned int paddedSizeOut, float* out, const float* A,
               const float* B,
               const float* C, unsigned int M, unsigned int N,
               unsigned int paddedN,
               unsigned int K, unsigned int paddedK)
{
    const auto strideA = M * paddedK;
    const auto strideB = K * paddedN;
    const auto strideC = M * paddedN;
    const auto strideOut = M * paddedN;

    for (long chunkIdx = 0; chunkIdx < static_cast<long>(
                                paddedSizeOut / strideOut); ++chunkIdx)
        for (size_t mIdx = 0; mIdx < M; ++mIdx)
            for (size_t nIdx = 0; nIdx < N; ++nIdx)
            {
                auto* batchPtrA = A + static_cast<size_t>(strideA) * chunkIdx;
                auto* batchPtrB = B + static_cast<size_t>(strideB) * chunkIdx;
                auto* batchPtrC = C + static_cast<size_t>(strideC) * chunkIdx;
                auto* batchPtrOut =
                    out + static_cast<size_t>(strideOut) * chunkIdx;

                float sum = batchPtrC[paddedN * mIdx + nIdx];
                for (size_t kIdx = 0; kIdx < K; ++kIdx)
                    sum += batchPtrA[paddedK * mIdx + kIdx] *
                        batchPtrB[paddedN * kIdx + nIdx];

                batchPtrOut[paddedN * mIdx + nIdx] = sum;
            }
}

void Gemm(float* out, float* A, float* B, float* C, unsigned int M,
          unsigned int N, unsigned int paddedN, unsigned int K,
          unsigned int paddedK, unsigned int batchSizeOut,
          unsigned int batchSizeA, unsigned int batchSizeB,
          unsigned int batchSizeC, unsigned int unitBatchSize)
{
    const auto strideA = M * paddedK;
    const auto strideB = K * paddedN;
    const auto strideC = M * paddedN;
    const auto strideOut = M * paddedN;

    for (size_t unitBatchIdx = 0; unitBatchIdx < batchSizeOut;
         unitBatchIdx += unitBatchSize)
        for (size_t batchIdx = unitBatchIdx;
             batchIdx < unitBatchIdx + unitBatchSize; ++batchIdx)
            for (size_t mIdx = 0; mIdx < M; ++mIdx)
                for (size_t nIdx = 0; nIdx < N; ++nIdx)
                {
                    auto* batchPtrA = A + strideA * (batchIdx % batchSizeA);
                    auto* batchPtrB = B + strideB * (batchIdx % batchSizeB);
                    auto* batchPtrC = C + strideC * (batchIdx % batchSizeC);
                    auto* batchPtrOut =
                        out + strideOut * (batchIdx % batchSizeOut);

                    float sum = batchPtrC[paddedN * mIdx + nIdx];
                    for (size_t kIdx = 0; kIdx < K; ++kIdx)
                        sum += batchPtrA[paddedK * mIdx + kIdx] *
                            batchPtrB[paddedN * kIdx + nIdx];

                    batchPtrOut[paddedN * mIdx + nIdx] = sum;
                }
}
} // namespace Sapphire::Compute::Naive::Dense
