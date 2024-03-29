// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/kernels/CrossEntropyKernel.cuh>

#define MIN_FLOAT 1.17549e-30f

namespace Sapphire::Compute::Dense::Cuda
{
__global__ void CrossEntropyKernel(float* y, const float* x, const float* label,
                                   int batchSize, int unitSize)
{
    const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId >= batchSize)
        return;

    float sum = 0.0f;
    for (int j =0; j < unitSize; ++j)
    {
        const auto idx = threadId * unitSize + j;
        const auto val = x[idx] == 0.0f ? MIN_FLOAT : x[idx];
        sum -= label[idx] * logf(val);
    }
    y[threadId] = sum;
}

__global__ void CrossEntropyBackwardKernel(float* dx, const float* x, const float* label,
                                           int batchSize, int unitSize)
{
    const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId >= batchSize)
        return;

    for (int j = 0; j < unitSize; ++j)
    {
        const auto idx = threadId * unitSize + j;
        const auto val = x[idx] == 0.0f ? MIN_FLOAT : x[idx];
        dx[idx] -= label[idx] / val;
    }
}
}
