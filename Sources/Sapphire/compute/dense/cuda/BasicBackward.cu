// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <Sapphire/compute/dense/cuda/kernels/BasicBackwardKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void DotBackward(unsigned int totalSize, float* da, float* db,
                          const float* dy, const float* a, const float* b,
                          unsigned inputStride, bool
                          broadcastInputA, bool broadcastInputB)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;
    const auto blockDim = totalSize / MAX_THREAD_DIM_X;
    const auto firstLaunchSize = blockDim * MAX_THREAD_DIM_X;

    if (firstLaunchSize > 0)
    {
        DotBackwardKernel<<<blockDim, threadDim>>>(
            da, db, dy, a, b, 0, firstLaunchSize, totalSize, inputStride,
            broadcastInputA, broadcastInputB);
    }

    if (totalSize > firstLaunchSize)
    {
        const unsigned int offset = firstLaunchSize;
        DotBackwardKernel<<<blockDim, threadDim>>>(
            da, db, dy, a, b, offset, totalSize - firstLaunchSize, totalSize,
            inputStride,
            broadcastInputA, broadcastInputB);
    }
}

__host__ void PowBackward(float* dx, const float* dy, const float* x,
                          const float factor, unsigned totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        PowBackwardKernel<<<blockDim, threadDim>>>(dx, dy, x, factor,
                                                   firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* xOffset = x + firstLaunchSize;
        const float* dyOffset = dy + firstLaunchSize;
        float* dxOffset = dx + firstLaunchSize;

        PowBackwardKernel<<<blockDim, threadDim>>>(
            dxOffset, dyOffset, xOffset, factor, totalSize - firstLaunchSize);
    }
}
}
