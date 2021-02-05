// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/naive/NaiveGemm.hpp>
#include <cstdlib>

namespace Motutapu::Compute::Naive::Dense
{
void NaiveGemm(float* out, float* A, float* B, float* C, unsigned int M,
               unsigned int N, unsigned int paddedN, unsigned int K,
               unsigned int paddedK, unsigned int batchSize, bool broadcastA,
               bool broadcastB, bool broadcastC)
{
    const auto strideA = (broadcastA ? 0 : (M * paddedK));
    const auto strideB = (broadcastB ? 0 : (K * paddedN));
    const auto strideC = (broadcastC ? 0 : (M * paddedN));

    //#pragma omp parallel for collapse(3) schedule(static)
    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (size_t mIdx = 0; mIdx < M; ++mIdx)
            for (size_t nIdx = 0; nIdx < N; ++nIdx)
            {
                auto* batchPtrA = A + batchIdx * strideA;
                auto* batchPtrB = B + batchIdx * strideB;
                auto* batchPtrC = C + batchIdx * strideC;
                auto* batchPtrOut = out + batchIdx * (M * paddedN);
                float sum = batchPtrC[paddedN * mIdx + nIdx];
                for (size_t kIdx = 0; kIdx < K; ++kIdx)
                    sum += batchPtrA[paddedK * mIdx + kIdx] *
                           batchPtrB[paddedN * kIdx + nIdx];

                batchPtrOut[paddedN * mIdx + nIdx] = sum;
            }
}
}  // namespace Motutapu::Compute::Naive::Dense
