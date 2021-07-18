// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/Trigonometric.cuh>
#include <Sapphire/compute/dense/cuda/kernels/TrigonometricKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void Cos(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        CosKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        CosKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void Sin(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        SinKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        SinKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void Tan(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        TanKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        TanKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void Cosh(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        CoshKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        CosKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void Sinh(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        SinhKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        SinhKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void Tanh(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        TanhKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        TanhKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void ArcCos(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        ArcCosKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        ArcCosKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void ArcSin(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        ArcSinKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        ArcSinKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void ArcTan(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        ArcTanKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        ArcTanKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void ArcCosh(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        ArcCoshKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        ArcCoshKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void ArcSinh(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        ArcSinhKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        ArcSinhKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void ArcTanh(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        ArcTanhKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        ArcTanhKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}
}
