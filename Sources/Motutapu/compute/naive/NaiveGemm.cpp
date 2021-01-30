// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/naive/NaiveGemm.hpp>
#include <cstdlib>

namespace Motutapu::Compute::Naive::Dense
{
void NaiveGemm(float* out, float* A, float* B, float* C,
                      unsigned int paddedM, unsigned int paddedN,
                      unsigned int paddedK, unsigned int batchSize,
                      bool broadcastA, bool broadcastB, bool broadcastC)
{
    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto* batchPtrA = A + batchIdx * (broadcastA ? 1 : (paddedM * paddedK));
        auto* batchPtrB = B + batchIdx * (broadcastB ? 1 : (paddedK * paddedN));
        auto* batchPtrC = C + batchIdx * (broadcastC ? 1 : (paddedM * paddedN));
        auto* batchPtrOut = out + batchIdx * (paddedM * paddedN);

        for (size_t mIdx = 0; mIdx < paddedM; ++mIdx)
            for (size_t nIdx = 0; nIdx < paddedN; ++nIdx)
            {
                float sum = 0.0f;
                for (size_t kIdx = 0; kIdx < paddedK; ++kIdx)
                    sum += batchPtrA[paddedK * mIdx + kIdx] *
                           batchPtrB[paddedN * kIdx + nIdx];

                batchPtrOut[paddedN * mIdx + nIdx] =
                    sum + batchPtrC[paddedM * mIdx + nIdx];
            }
    }
}
}  // namespace Motutapu::Compute::Naive::Dense
