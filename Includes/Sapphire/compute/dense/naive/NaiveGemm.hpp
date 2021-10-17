// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_NAIVEGEMM_HPP
#define Sapphire_COMPUTE_NAIVEGEMM_HPP

namespace Sapphire::Compute::Dense::Naive
{
void Gemm(unsigned int totalSize, float* out, const float* A, const float* B,
          unsigned int M, unsigned int N,
          unsigned int K);
} // namespace Sapphire::Compute::Naive::Dense

#endif
