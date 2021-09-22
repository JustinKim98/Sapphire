// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/kernels/BasicBackwardKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__global__ void DotBackwardKernel(float* da, float* db, const float* dy,
                                  const float* a, const float* b,
                                  unsigned int offset, unsigned int launchSize,
                                  unsigned int totalSize,
                                  unsigned int inputStride,
                                  bool broadcastInputA, bool broadcastInputB)
{
    const auto sizePerBlock = launchSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    const unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    const unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = offset + blockOffset + blockDim.x * i + threadIdx.x;
        da[idx % leftOverA] += dy[idx] * b[idx % leftOverB];
        db[idx % leftOverB] += dy[idx] * a[idx % leftOverA];
    }
}

__global__ void PowBackwardKernel(float* dx, const float* dy, const float* x,
                                  const float factor, unsigned totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] * factor * powf(x[idx], factor - 1);
    }
}

__global__ void MeanBackwardKernel(float* dx, const float* x, const float* dy,
    unsigned int yTotalSize,
                                   unsigned int unitSize, unsigned int stride)
{
    const auto unitId = blockIdx.x * blockDim.x + threadIdx.x;
    const auto outerId = unitId / stride;
    const auto innerId = unitId % stride;

    if (unitId < yTotalSize)
    {
        for (unsigned int i = 0; i < unitSize; ++i)
            dx[unitSize * stride * outerId + i * stride + innerId] +=dy[unitId]/((float) unitSize);
    }
}

}  // namespace Sapphire::Compute::Dense::Cuda
