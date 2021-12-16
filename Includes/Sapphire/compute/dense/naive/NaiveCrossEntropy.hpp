// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_NAIVE_CROSS_ENTROPY_HPP
#define SAPPHIRE_COMPUTE_DENSE_NAIVE_CROSS_ENTROPY_HPP

namespace Sapphire::Compute::Dense::Naive
{
void CrossEntropy(float* y, const float* x, const float* label, int batchSize,
                  int unitSize);

void CrossEntropyBackward(float* dx, const float* x, const float* label,
                          int batchSize, int unitSize);
}

#endif
