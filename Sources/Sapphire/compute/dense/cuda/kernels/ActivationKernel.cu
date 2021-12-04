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
    const auto batchId = unitId / unitSize;
    const auto i = unitId % unitSize;
    const auto iIdx = unitSize * batchId + i;

    if (unitId < totalSize)
    {
        float max = 0.0f;
        for(int j = 0; j < unitSize; ++j)
        {
            const auto jIdx = unitSize * batchId + j;
            const auto data = x[jIdx];
            if (data > max)
                max = data;
        }

        float sum = 0;
        for (unsigned int j = 0; j < unitSize; j++)
        {
            const auto jIdx = unitSize * batchId + j;
            sum += expf(x[jIdx] - max);
        }
        y[iIdx] = expf(x[iIdx] - max) / sum;
    }
}
}
