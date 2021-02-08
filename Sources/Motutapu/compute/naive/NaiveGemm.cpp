// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/naive/NaiveGemm.hpp>
#include <cstdlib>
#include <iostream>

namespace Motutapu::Compute::Naive::Dense
{
void NaiveGemm(unsigned int offsetOut, unsigned int offsetA,
               unsigned int offsetB, unsigned int offsetC, float* out, float* A,
               float* B, float* C, unsigned int M, unsigned int N,
               unsigned int paddedN, unsigned int K, unsigned int paddedK,
               unsigned int totalSizeOut)
{
    const auto strideA = M * paddedK;
    const auto strideB = K * paddedN;
    const auto strideC = M * paddedN;
    const auto strideOut = M * paddedN;

    for (size_t batchIdx = 0; batchIdx < totalSizeOut / strideOut; ++batchIdx)
        for (size_t mIdx = 0; mIdx < M; ++mIdx)
            for (size_t nIdx = 0; nIdx < N; ++nIdx)
            {
                auto* batchPtrA = A + offsetA + strideA * batchIdx;
                auto* batchPtrB = B + offsetB + strideB * batchIdx;
                auto* batchPtrC = C + offsetC + strideC * batchIdx;
                auto* batchPtrOut = out + offsetOut + strideOut * batchIdx;

                float sum = batchPtrC[paddedN * mIdx + nIdx];
                for (size_t kIdx = 0; kIdx < K; ++kIdx)
                    sum += batchPtrA[paddedK * mIdx + kIdx] *
                           batchPtrB[paddedN * kIdx + nIdx];

                batchPtrOut[paddedN * mIdx + nIdx] = sum;
                //std::cout << "sum : " << sum << std::endl;
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
}  // namespace Motutapu::Compute::Naive::Dense
