// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_NAIVEGEMM_HPP
#define MOTUTAPU_COMPUTE_NAIVEGEMM_HPP

namespace Motutapu::Compute::Naive::Dense
{
void NaiveGemm(unsigned int offsetOut, unsigned int offsetA,
               unsigned int offsetB, unsigned int offsetC, float* out, float* A,
               float* B, float* C, unsigned int M, unsigned int N,
               unsigned int paddedN, unsigned int K, unsigned int paddedK,
               unsigned int totalSizeOut);
}  // namespace Motutapu::Compute::Naive::Dense

#endif
