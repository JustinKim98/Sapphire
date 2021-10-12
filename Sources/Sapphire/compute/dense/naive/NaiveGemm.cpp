// Copyright (c) 2020, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/naive/NaiveGemm.hpp>
#include <cstdlib>

namespace Sapphire::Compute::Dense::Naive
{
void Gemm(unsigned int totalSize, float* out, const float* A,
               const float* B,
               const float* C, unsigned int M, unsigned int N,
               unsigned int K)
{
    const auto strideA = M * K;
    const auto strideB = K * N;
    const auto strideC = M * N;
    const auto strideOut = M * N;

    for (long chunkIdx = 0; chunkIdx < static_cast<long>(
                                totalSize / strideOut); ++chunkIdx)
        for (size_t mIdx = 0; mIdx < M; ++mIdx)
            for (size_t nIdx = 0; nIdx < N; ++nIdx)
            {
                auto* batchPtrA =
                    A + static_cast<std::size_t>(strideA) * chunkIdx;
                auto* batchPtrB =
                    B + static_cast<std::size_t>(strideB) * chunkIdx;
                auto* batchPtrC =
                    C + static_cast<std::size_t>(strideC) * chunkIdx;
                auto* batchPtrOut =
                    out + static_cast<std::size_t>(strideOut) * chunkIdx;

                float sum = batchPtrC[N * mIdx + nIdx];
                for (size_t kIdx = 0; kIdx < K; ++kIdx)
                    sum += batchPtrA[K * mIdx + kIdx] *
                        batchPtrB[N * kIdx + nIdx];
                batchPtrOut[N * mIdx + nIdx] = sum;
            }
}
} // namespace Sapphire::Compute::Naive::Dense
