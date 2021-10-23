// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_NAIVEBASIC_HPP
#define Sapphire_NAIVEBASIC_HPP

namespace Sapphire::Compute::Dense::Naive
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

void Transpose(float* output, const float* input, unsigned int inputRows,
               unsigned int inputCols,
               unsigned int batchSize,
               bool broadcast);

void Pow(float* output, const float* input, float exponent,
         unsigned int totalSize);

void Cos(float* output, const float* input, unsigned int totalSize);

void Sin(float* output, const float* input, unsigned int totalSize);

void Tan(float* output, const float* input, unsigned int totalSize);

void Cosh(float* output, const float* input, unsigned int totalSize);

void Sinh(float* output, const float* input, unsigned int totalSize);

void Tanh(float* output, const float* input, unsigned int totalSize);

void log(float* output, const float* input, unsigned int totalSize);

void log10(float* output, const float* input, unsigned int totalSize);

void ReLU(float* output, const float* input, unsigned int totalSize);

void ReLUBackward(float* dx, const float* dy, const float* x,
                  unsigned int totalSize);

void LeakyReLU(float* output, const float* input, float a,
               unsigned int totalSize);

void LeakyReLUBackward(float* output, const float* input, float a,
                       unsigned int totalSize);

void Inverse(float* output, const float* input, unsigned int totalSize);

void Mean(float* y, const float* x,
          unsigned ySize, unsigned int unitSize, unsigned stride);

void MeanBackward(float* dx, const float* dy,
                  unsigned int ySize, unsigned int unitSize,
                  unsigned int stride);

void Softmax(float* output, const float* input, unsigned int totalSize,
             unsigned int unitSize);

void SoftmaxBackward(float* dx, const float* dy, const float* y,
                     unsigned int totalSize, unsigned int unitSize);
} // namespace Sapphire::Compute::Naive::Dense

#endif  // Sapphire_NAIVEBASIC_HPP
