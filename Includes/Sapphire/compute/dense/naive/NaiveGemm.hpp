// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_NAIVEGEMM_HPP
#define Sapphire_COMPUTE_NAIVEGEMM_HPP

namespace Sapphire::Compute::Naive::Dense
{
void NaiveGemm(unsigned int totalSize, float* out, float* A, float* B, float* C,
               unsigned int M, unsigned int N, unsigned int paddedN,
               unsigned int K, unsigned int paddedK);
}  // namespace Sapphire::Compute::Naive::Dense

#endif
