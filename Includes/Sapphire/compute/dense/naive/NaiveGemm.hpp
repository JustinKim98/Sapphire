// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_NAIVEGEMM_HPP
#define Sapphire_COMPUTE_NAIVEGEMM_HPP

namespace Sapphire::Compute::Dense::Naive
{
void NaiveGemm(unsigned int paddedSizeOut, float* out, const float* A, const float* B, const
               float* C,
               unsigned int M, unsigned int N, unsigned int paddedN,
               unsigned int K, unsigned int paddedK);

void Gemm(float* out, float* A, float* B, float* C, unsigned int M,
          unsigned int N, unsigned int paddedN, unsigned int K,
          unsigned int paddedK, unsigned int batchSizeOut,
          unsigned int batchSizeA, unsigned int batchSizeB,
          unsigned int batchSizeC, unsigned int unitBatchSize);
}  // namespace Sapphire::Compute::Naive::Dense

#endif
