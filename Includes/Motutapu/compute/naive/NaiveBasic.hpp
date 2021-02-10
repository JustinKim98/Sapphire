// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_NAIVEBASIC_HPP
#define MOTUTAPU_NAIVEBASIC_HPP

namespace Motutapu::Compute::Naive::Dense
{
void Add(unsigned int totalSize, float* output, const float* inputA,
         const float* inputB, unsigned int inputStride, bool broadcastInputA,
         bool broadcastInputB);

void Sub(unsigned int totalSize, float* output, const float* inputA,
         const float* inputB, unsigned int inputStride, bool broadcastInputA,
         bool broadcastInputB);

void Dot(unsigned int totalSize, float* output, const float* inputA,
         const float* inputB, unsigned int inputStride, bool broadcastInputA,
         bool broadcastInputB);

void Scale(float* output, const float* input, float scaleFactor,
           unsigned int totalSize);

void Transpose(float* output, const float* input, unsigned int inputNumRows,
               unsigned int inputNumCols, unsigned int batchSize,
               bool broadcast);

}  // namespace Motutapu::Compute::Naive::Dense

#endif  // MOTUTAPU_NAIVEBASIC_HPP
