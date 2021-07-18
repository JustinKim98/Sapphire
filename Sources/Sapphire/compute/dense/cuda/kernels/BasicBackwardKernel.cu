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

__global__ void CosBackwardKernel(float* dx, const float* dy, const float* x,
                                  unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] -= dy[idx] * sinf(x[idx]);
    }
}

__global__ void SinBackwardKernel(float* dx, const float* dy, const float* x,
                                  unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] * cosf(x[idx]);
    }
}

__global__ void TanBackwardKernel(float* dx, const float* dy, const float* x,
                                  unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] -= dy[idx] * powf((1 / cosf(x[idx])), 2);
    }
}

__global__ void CoshBackwardKernel(float* dx, const float* dy, const float* x,
                                   unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] * sinhf(x[idx]);
    }
}

__global__ void SinhBackwardKernel(float* dx, const float* dy, const float* x,
                                   unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] * coshf(x[idx]);
    }
}

__global__ void TanhBackwardKernel(float* dx, const float* dy, const float* x,
                                   unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] / powf(coshf(x[idx]), 2);
    }
}

__global__ void ArcCosBackwardKernel(float* dx, const float* dy, const float* x,
                                     unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] -= dy[idx] / (1 - powf(x[idx], 2));
    }
}

__global__ void ArcSinBackwardKernel(float* dx, const float* dy, const float* x,
                                     unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] / (1 - powf(x[idx], 2));
    }
}

__global__ void ArcTanBackwardKernel(float* dx, const float* dy, const float* x,
                                     unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] / (1 + powf(x[idx], 2));
    }
}

__global__ void ArcCoshBackwardKernel(float* dx, const float* dy,
                                      const float* x,
                                      unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] / (powf(x[idx], 2) - 1);
    }
}

__global__ void ArcSinhBackwardKernel(float* dx, const float* dy,
                                      const float* x,
                                      unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] / (1 + powf(x[idx], 2));
    }
}

__global__ void ArcTanhBackwardKernel(float* dx, const float* dy,
                                      const float* x,
                                      unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        dx[idx] += dy[idx] / (1 - powf(x[idx], 2));
    }
}
}
