// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#define MIN_FLOAT 1.17549e-30f

#include <Sapphire/compute/dense/naive/NaiveCrossEntropy.hpp>
#include <cmath>
#include <iostream>

namespace Sapphire::Compute ::Dense::Naive
{
void CrossEntropy(float* y, const float* x, const float* label, int batchSize,
                  int unitSize)
{
    for (int i = 0; i < batchSize; ++i)
    {
        float sum = 0.0f;
        for (int j = 0; j < unitSize; ++j)
        {
            const auto idx = i * unitSize + j;
            const auto val = x[idx] == 0.0f
                                 ? MIN_FLOAT
                                 : x[idx];
            sum -= label[idx] * logf(val);
        }
        y[i] = sum;
    }
}

void CrossEntropyBackward(float* dx, const float* x, const float* label,
                          int batchSize,
                          int unitSize)
{
    for (int i = 0; i < batchSize; ++i)
    {
        for (int j = 0; j < unitSize; ++j)
        {
            const auto idx = i * unitSize + j;
            const auto val = x[idx] == 0.0f ? MIN_FLOAT : x[idx];
            const auto outData = label[idx] / val;
            if (std::isnan(outData))
            {
                std::cout << "nan detected" << std::endl;
            }
            dx[idx] -= outData;
        }
    }
}
}
