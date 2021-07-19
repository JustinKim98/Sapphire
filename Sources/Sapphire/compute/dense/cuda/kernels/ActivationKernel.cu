// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/kernels/ActivationKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__global__ void ReLUKernel(float* y, const float* x, unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

__global__ void LeakyReLUKernel(float* y, const float* x,
                                float a, unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[idx] = x[idx] > 0 ? x[idx] : a * x[idx];
    }
}

__global__ void SoftMaxKernel(float* y, const float* x, unsigned int totalSize,
                              unsigned int unitSize)
{
    const auto unitId = blockIdx.x * blockDim.x + threadIdx.x;
    const auto curBatch = unitId / blockDim.x;
    const auto curIdx = unitId % unitSize;

    if (unitId < totalSize)
    {
        float sum = 0;
        for (unsigned int i = 0; i < unitSize; i++)
        {
            sum += expf(x[unitSize * curBatch + i]);
        }
        y[unitSize * curBatch + curIdx] =
            expf(x[unitSize * curBatch + curIdx]) / sum;
    }
}
}