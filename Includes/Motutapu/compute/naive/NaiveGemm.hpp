// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_NAIVEGEMM_HPP
#define MOTUTAPU_COMPUTE_NAIVEGEMM_HPP

namespace Motutapu::Compute::Naive::Dense
{
void NaiveGemm(float* out, float* A, float* B, float* C,
                      unsigned int paddedM, unsigned int paddedN,
                      unsigned int paddedK, unsigned int batchSize,
                      bool broadcastA, bool broadcastB, bool broadcastC);
}  // namespace Motutapu::Compute::Naive::Dense

#endif
