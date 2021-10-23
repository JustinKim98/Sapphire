// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/kernels/ActivationBackwardKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__global__ void ReLUBackwardKernel(float* dx, const float* dy,
                                   const float* x, unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;

        dx[idx] = x[idx] > 0.0f ? dy[idx] : 0.0f;
    }
}


__global__ void LeakyReLUBackwardKernel(float* dx, const float* dy,
                                        const float* x, const float a,
                                        unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] = x[idx] > 0.0f ? dy[idx] : a;
    }
}

__global__ void SoftMaxBackwardKernel(float* dx, const float* dy,
                                      const float* y,
                                      unsigned int totalSize,
                                      unsigned int unitSize)
{
    const auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const auto batchId = threadId / unitSize;
    const auto i = threadId % unitSize;
    const auto iIdx = unitSize * batchId + i;

    if (threadId < totalSize)
    {
        float sum = 0;
        for (unsigned int j = 0; j < unitSize; j++)
        {
            const auto jIdx = unitSize * batchId + j;
            if (j == i)
                sum += dy[jIdx] * (y[jIdx] * (1.0f - y[jIdx]));
            else
                sum += dy[jIdx] * (-y[iIdx] * y[jIdx]);
        }
        dx[iIdx] = sum;
    }
}
}
