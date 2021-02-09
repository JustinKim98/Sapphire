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

void Dot(float* output, const float* inputA, const float* inputB,
         unsigned int totalSize, unsigned int inputStride, bool broadcastInputA,
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
           unsigned int totalSize, unsigned int inputStride,
           bool broadcastInput)
{
    unsigned int leftOver = broadcastInput ? inputStride : totalSize;

    for (unsigned int i = 0; i < totalSize; i++)
    {
        output[i] = input[i % leftOver] * scaleFactor;
    }
}

void Transpose(float* output, const float* input, unsigned int inputNumRows,
               unsigned int inputNumCols, unsigned int batchSize,
               bool broadcast)
{
    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
        for (unsigned int i = 0; i < inputNumRows; i++)
            for (unsigned int j = 0; j < inputNumCols; j++)
            {
                float* outputOffset =
                    output + batchIdx * inputNumRows * inputNumCols;
                const float* inputOffset =
                    input +
                    (broadcast ? 0 : batchIdx * inputNumRows * inputNumCols);
                outputOffset[j * inputNumRows + i] =
                    inputOffset[i * inputNumCols + j];
            }
}
}  // namespace Motutapu::Compute::Naive::Dense