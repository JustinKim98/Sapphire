// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/naive/NaiveBasic.hpp>
#include <cmath>

namespace Sapphire::Compute::Dense::Naive
{
void Add(unsigned int totalSize, float* output, const float* inputA,
         const float* inputB, unsigned int inputStride, bool broadcastInputA,
         bool broadcastInputB)
{
    const unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    const unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < totalSize; i++)
    {
        output[i] = inputA[i % leftOverA] + inputB[i % leftOverB];
    }
}

void Sub(unsigned int totalSize, float* output, const float* inputA,
         const float* inputB, unsigned int inputStride, bool broadcastInputA,
         bool broadcastInputB)
{
    const unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    const unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < totalSize; i++)
    {
        output[i] = inputA[i % leftOverA] - inputB[i % leftOverB];
    }
}

void Dot(unsigned int totalSize, float* output, const float* inputA,
         const float* inputB, unsigned int inputStride, bool broadcastInputA,
         bool broadcastInputB)
{
    const unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    const unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

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
               unsigned int inputCols,
               unsigned int batchSize,
               bool broadcast)
{
    const auto leftOver = broadcast
                              ? inputRows * inputCols
                              : batchSize * inputRows * inputCols;
    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
        for (unsigned int i = 0; i < inputRows; i++)
            for (unsigned int j = 0; j < inputCols; j++)
            {
                float* outputOffset = output +
                                      static_cast<std::size_t>(batchIdx) *
                                      inputCols * inputRows;
                const float* inputOffset =
                    input + static_cast<std::size_t>(batchIdx) * inputRows *
                    inputCols;
                outputOffset[j * inputRows + i] =
                    inputOffset[(i * inputCols + j) % leftOver];
            }
}

void Pow(float* output, const float* input, const float exponent,
         unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = std::pow(input[i], exponent);
    }
}

void Cos(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = std::cos(input[i]);
    }
}

void Sin(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = std::sin(input[i]);
    }
}

void Tan(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = std::tan(input[i]);
    }
}

void Cosh(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = std::cosh(input[i]);
    }
}

void Sinh(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = std::sinh(input[i]);
    }
}

void Tanh(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = std::tanh(input[i]);
    }
}

void log(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = std::log(input[i]);
    }
}

void log10(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = std::log10(input[i]);
    }
}

void ReLU(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

void ReLUBackward(float* dx, const float* dy, const float* x,
                  unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
    }
}

void LeakyReLU(float* output, const float* input, float a,
               unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = input[i] > 0 ? input[i] : a * input[i];
    }
}

void LeakyReLUBackward(float* output, const float* input, float a,
                       unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = input[i] > 0 ? 1 : a;
    }
}

void Inverse(float* output, const float* input, unsigned int totalSize)
{
    for (unsigned int i = 0; i < totalSize; ++i)
    {
        output[i] = 1.0f / input[i];
    }
}

void Mean(float* y, const float* x,
          unsigned ySize, unsigned int unitSize, unsigned stride)
{
    for (unsigned int unitId = 0; unitId < ySize; unitId++)
    {
        const auto outerId = unitId / stride;
        const auto innerId = unitId % stride;

        for (unsigned int i = 0; i < unitSize; i++)
        {
            y[unitId] += x[unitSize * stride * outerId + i * stride + innerId];
        }
        y[unitId] /= static_cast<float>(unitSize);
    }
}

void MeanBackward(float* dx, const float* dy,
                  unsigned int ySize, unsigned int unitSize,
                  unsigned int stride)
{
    for (unsigned int unitId = 0; unitId < ySize; unitId++)
    {
        const auto outerId = unitId / stride;
        const auto innerId = unitId % stride;

        for (unsigned int i = 0; i < unitSize; i++)
        {
            dx[unitSize * stride * outerId + i * stride + innerId] +=
                dy[unitId] / static_cast<float>(unitSize);
        }
    }
}

void Softmax(float* output, const float* input, unsigned int totalSize,
             unsigned int unitSize)
{
    const auto batchSize = totalSize / unitSize;

    for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        float sum = 0;
        for (unsigned int i = 0; i < unitSize; ++i)
            sum += std::exp(input[unitSize * batchIdx + i]);

        for (unsigned int i = 0; i < unitSize; ++i)
            output[unitSize * batchIdx + i] =
                std::exp(input[unitSize * batchIdx + i]) / sum;
    }
}

void SoftmaxBackward(float* dx, const float* dy, const float* x,
                     unsigned int totalSize, unsigned int unitSize)
{
    const auto batchSize = totalSize / unitSize;

    for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const unsigned int offset = unitSize * batchIdx;
        float sum = 0;
        for (unsigned int unitIdx = 0; unitIdx < unitSize; ++unitIdx)
            for (unsigned int i = 0; i < unitSize; ++i)
            {
                if (i == unitIdx)
                {
                    dx[offset + i] += dy[offset + i] *
                        (x[offset + i] * (1 - x[offset + i]));
                }
                else
                {
                    sum += dy[offset + i] *
                        (-x[offset + unitIdx] * x[offset + i]);
                }
            }
    }
}
} // namespace Sapphire::Compute::Naive::Dense
