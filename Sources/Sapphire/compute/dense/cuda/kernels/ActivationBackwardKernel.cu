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
    const auto unitId = blockIdx.x * blockDim.x + threadIdx.x;
    const auto curUnitIdx = unitId / unitSize;
    const auto i = curUnitIdx * unitSize + unitId % unitSize;

    if (i < totalSize)
    {
        float gradX = 0;
        for (unsigned int idx = 0; idx < unitSize; idx++)
        {
            const auto j = unitSize * curUnitIdx + idx;
            if (j == i)
                gradX += dy[j] * (y[j] * (1 - y[j]));
            else
                gradX += dy[j] * (-y[i] * y[j]);
        }
        dx[i] = gradX;
    }
}
}
