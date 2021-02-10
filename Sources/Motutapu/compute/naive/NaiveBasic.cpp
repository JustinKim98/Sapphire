// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/naive/NaiveBasic.hpp>

namespace Motutapu::Compute::Naive::Dense
{
void Add(unsigned int totalSize, float* output, const float* inputA,
         const float* inputB, unsigned int inputStride, bool broadcastInputA,
         bool broadcastInputB)
{
    unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < totalSize; i++)
    {
        output[i] = inputA[i % leftOverA] + inputB[i % leftOverB];
    }
}

void Sub(unsigned int totalSize, float* output, const float* inputA,
         const float* inputB, unsigned int inputStride, bool broadcastInputA,
         bool broadcastInputB)
{
    unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < totalSize; i++)
    {
        output[i] = inputA[i % leftOverA] - inputB[i % leftOverB];
    }
}

void Dot(unsigned int totalSize, float* output, const float* inputA,
         const float* inputB, unsigned int inputStride, bool broadcastInputA,
         bool broadcastInputB)
{
    unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < totalSize; i++)
    {
        output[i] = inputA[i % leftOverA] * inputB[i % leftOverB];
    }
}

void Scale(float* output, const float* input, const float scaleFactor,
           unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; i++)
    {
        output[i] = input[i] * scaleFactor;
    }
}

void Transpose(float* output, const float* input, unsigned int inputRows,
               unsigned int paddedInputRows, unsigned int inputCols,
               unsigned int paddedInputCols, unsigned int batchSize,
               bool broadcast)
{
    const auto leftOver = broadcast ? inputRows * paddedInputCols
                                    : batchSize * inputRows * paddedInputCols;
    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
        for (unsigned int i = 0; i < inputRows; i++)
            for (unsigned int j = 0; j < inputCols; j++)
            {
                float* outputOffset =
                    output + batchIdx * inputCols * paddedInputRows;
                const float* inputOffset =
                    input + batchIdx * inputRows * paddedInputCols;
                outputOffset[j * paddedInputRows + i] =
                    inputOffset[(i * paddedInputCols + j) % leftOver];
            }
}
}  // namespace Motutapu::Compute::Naive::Dense