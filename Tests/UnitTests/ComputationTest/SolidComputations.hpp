// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TEST_SOLIDCOMPUTATIONS_HPP
#define MOTUTAPU_TEST_SOLIDCOMPUTATIONS_HPP

#include <Motutapu/tensor/TensorData.hpp>

namespace Motutapu::Test::Solid
{
template <typename T>
void Gemm(T* out, T* A, T* B, T* C, unsigned int paddedM,
          unsigned int paddedN, unsigned int paddedK,
          unsigned int batchSize, bool broadcastA, bool broadcastB,
          bool broadcastC)
{
    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto* batchPtrA =
            A + batchIdx * (broadcastA ? 1 : (paddedM * paddedK));
        auto* batchPtrB =
            B + batchIdx * (broadcastB ? 1 : (paddedK * paddedN));
        auto* batchPtrC =
            C + batchIdx * (broadcastC ? 1 : (paddedM * paddedN));
        auto* batchPtrOut = out + batchIdx * (paddedM * paddedN);

        for (std::size_t mIdx = 0; mIdx < paddedM; ++mIdx)
            for (std::size_t nIdx = 0; nIdx < paddedN; ++nIdx)
            {
                T sum = static_cast<T>(0);
                for (std::size_t kIdx = 0; kIdx < paddedK; ++kIdx)
                    sum += batchPtrA[paddedK * mIdx + kIdx] * batchPtrB[
                        paddedN * kIdx + nIdx];

                batchPtrOut[paddedN * mIdx + nIdx] =
                    sum + batchPtrC[paddedM * mIdx + nIdx];;
            }
    }
}
}

#endif
