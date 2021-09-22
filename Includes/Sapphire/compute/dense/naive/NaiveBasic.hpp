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
           unsigned int totalSize, unsigned colSize, unsigned padSize);

void Transpose(float* output, const float* input, unsigned int inputRows,
               unsigned int paddedInputRows, unsigned int inputCols,
               unsigned int paddedInputCols, unsigned int batchSize,
               bool broadcast);

void Pow(float* output, const float* input, float exponent,
         unsigned int totalSize, unsigned colSize, unsigned padSize);

void cos(float* output, const float* input, unsigned int totalSize,
         unsigned colSize, unsigned
         padSize);

void sin(float* output, const float* input, unsigned int totalSize,
         unsigned colSize, unsigned
         padSize);

void tan(float* output, const float* input, unsigned int totalSize,
         unsigned colSize, unsigned
         padSize);

void cosh(float* output, const float* input, unsigned int totalSize,
          unsigned colSize, unsigned
          padSize);

void sinh(float* output, const float* input, unsigned int totalSize,
          unsigned colSize, unsigned
          padSize);

void tanh(float* output, const float* input, unsigned int totalSize,
          unsigned colSize, unsigned
          padSize);

void log(float* output, const float* input, unsigned int totalSize,
         unsigned colSize, unsigned
         padSize);

void log10(float* output, const float* input, unsigned int totalSize,
           unsigned colSize, unsigned
           padSize);

void ReLU(float* output, const float* input, unsigned int totalSize,
          unsigned colSize, unsigned
          padSize);

void ReLUBackward(float* dx, const float* dy, const float* x,
                  unsigned int totalSize, unsigned colSize, unsigned
                  padSize);

void LeakyReLU(float* output, const float* input, float a,
               unsigned int totalSize, unsigned colSize, unsigned padSize);

void LeakyReLUBackward(float* output, const float* input, float a,
                       unsigned int totalSize, unsigned colSize,
                       unsigned padSize);

void Inverse(float* output, const float* input, unsigned int totalSize,
             unsigned colSize, unsigned
             padSize);

void Mean(float* y, const float* x,
          unsigned ySize, unsigned int unitSize, unsigned stride,
          unsigned yCols, unsigned yPadSize, unsigned xCols, unsigned xPadSize);

void MeanBackward(float* dx, const float* x, const float* dy,
                  unsigned int ySize, unsigned int unitSize,
                  unsigned int stride, unsigned int yCols, unsigned int yPadSize, unsigned xCols, unsigned
                  xPadSize);

void Softmax(float* output, const float* input, unsigned int paddedTotalSize,
             unsigned int unitSize, unsigned int paddedUnitSize);

void SoftmaxBackward(float* dx, const float* dy, const float* x,
                     unsigned int totalSize, unsigned int unitSize,
                     unsigned int padSize);
} // namespace Sapphire::Compute::Naive::Dense

#endif  // Sapphire_NAIVEBASIC_HPP
