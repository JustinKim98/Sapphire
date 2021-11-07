// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_NAIVE_CROSS_ENTROPY_HPP
#define SAPPHIRE_COMPUTE_DENSE_NAIVE_CROSS_ENTROPY_HPP

#include <cmath>

namespace Sapphire::Compute::Dense::Naive
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
            sum -= x[idx] * logf(label[idx]);
        }
        y[i] = sum;
    }
}

void CrossEntropyBackward(float* dx, const float* label, int batchSize,
                          int unitSize)
{
    for (int i = 0; i < batchSize; ++i)
    {
        for (int j = 0; j < unitSize; ++j)
        {
            const auto idx = i * unitSize + j;
            dx[idx] -= logf(label[idx]);
        }
    }
}
}

#endif
