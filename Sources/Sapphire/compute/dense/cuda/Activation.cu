// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/kernels/ActivationKernel.cuh>
#include <Sapphire/compute/dense/cuda/kernels/ActivationBackwardKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void ReLU(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        ReLUKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        ReLUKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void LeakyReLU(float* y, const float* x, const float a,
                        unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        LeakyReLUKernel<<<blockDim, threadDim>>>(y, x, a, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        LeakyReLUKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, a, totalSize - firstLaunchSize);
    }
}

__host__ void SoftMax(float* y, const float* x, unsigned int totalSize,
                      unsigned int unitSize)
{
    const auto blockDim = (unitSize > 512) ? 512 : unitSize;
    const auto gridDim = (totalSize % blockDim == 0)
                             ? totalSize / blockDim
                             : totalSize / blockDim + 1;
    SoftMaxKernel<<<gridDim, blockDim>>>(y, x, totalSize, unitSize);
}

__host__ void ReLUBackward(float* dx, const float* dy, const float* x,
                           unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        ReLUBackwardKernel<<<blockDim, threadDim>>>(dx, dy, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        x += firstLaunchSize;
        dx += firstLaunchSize;
        dy += firstLaunchSize;

        ReLUBackwardKernel<<<1, totalSize - firstLaunchSize>>>(
            dx, dy, x, totalSize - firstLaunchSize);
    }
}

__host__ void LeakyReLUBackward(float* dx, const float* dy, const float* x,
                                const float a, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        LeakyReLUBackwardKernel<<<blockDim, threadDim>>>(dx, dy, x, a,
            firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        x += firstLaunchSize;
        dx += firstLaunchSize;

        LeakyReLUBackwardKernel<<<1, totalSize - firstLaunchSize>>>(
            dx, dy, x, a, totalSize - firstLaunchSize);
    }
}

__host__ void SoftmaxBackward(float* dx, const float* dy, const float* x,
                          unsigned int totalSize, unsigned int unitSize)
{
    auto blockDim = (unitSize > 64) ? 64 : unitSize;
    const auto gridDim = (totalSize % blockDim == 0)
                             ? totalSize / blockDim
                             : totalSize / blockDim + 1;
    SoftMaxBackwardKernel<<<gridDim, blockDim>>>(
        dx, dy, x, totalSize, unitSize);
}
}
