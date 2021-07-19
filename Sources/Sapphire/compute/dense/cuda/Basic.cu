// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cublas_v2.h>
#include <cudnn.h>
#include <Sapphire/compute/dense/cuda/Basic.cuh>
#include <Sapphire/compute/dense/cuda/kernels/BasicKernel.cuh>
#include <Sapphire/compute/dense/cuda/kernels/TrigonometricKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void Add(unsigned int totalSize, float* y, const float* a,
                  const float* b, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / MAX_THREAD_DIM_X;
    const auto firstLaunchSize = blockDim * MAX_THREAD_DIM_X;

    if (firstLaunchSize > 0)
        AddKernel<<<blockDim, threadDim>>>(
            y, a, b, 0, firstLaunchSize, totalSize, inputStride,
            broadcastInputA, broadcastInputB);

    if (totalSize > firstLaunchSize)
    {
        const unsigned int offset = firstLaunchSize;

        AddKernel<<<1, totalSize - firstLaunchSize>>>(
            y, a, b, offset, totalSize - firstLaunchSize,
            totalSize, inputStride, broadcastInputA, broadcastInputB);
    }
}

__host__ void Sub(unsigned int totalSize, float* y, const float* a,
                  const float* b, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / MAX_THREAD_DIM_X;
    const auto firstLaunchSize = blockDim * MAX_THREAD_DIM_X;

    if (firstLaunchSize > 0)
        SubKernel<<<blockDim, threadDim>>>(
            y, a, b, 0, firstLaunchSize, totalSize, inputStride,
            broadcastInputA, broadcastInputB);

    if (totalSize > firstLaunchSize)
    {
        const unsigned int offset = firstLaunchSize;

        SubKernel<<<1, totalSize - firstLaunchSize>>>(
            y, a, b, offset, totalSize - firstLaunchSize,
            totalSize, inputStride, broadcastInputA, broadcastInputB);
    }
}

__host__ void Scale(float* y, const float* x, const float scaleFactor,
                    unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        ScaleKernel<<<blockDim, threadDim>>>(y, x, scaleFactor,
                                             firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        ScaleKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, scaleFactor,
            totalSize - firstLaunchSize);
    }
}

__host__ void Transpose(float* y, const float* x,
                        unsigned int inputNumRows, unsigned int inputNumCols,
                        unsigned int batchSize, bool broadcastInput)
{
    const auto tileDim = 8;
    const unsigned int blockDimX = (inputNumCols % tileDim == 0)
                                       ? inputNumCols / tileDim
                                       : inputNumCols / tileDim + 1;
    const unsigned int blockDimY = (inputNumRows % tileDim == 0)
                                       ? inputNumRows / tileDim
                                       : inputNumRows / tileDim + 1;

    const unsigned int blockDimZ = batchSize;
    const dim3 blockDim(blockDimX, blockDimY, blockDimZ);
    const dim3 threadDim(tileDim, 8);
    TransposeKernel<<<blockDim, threadDim>>>(y, x, inputNumRows,
                                             inputNumCols, broadcastInput);
}

__host__ void Dot(unsigned int totalSize, float* y, const float* a,
                  const float* b, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / MAX_THREAD_DIM_X;
    const auto firstLaunchSize = blockDim * MAX_THREAD_DIM_X;

    if (firstLaunchSize > 0)
        DotKernel<<<blockDim, threadDim>>>(
            y, a, b, 0, firstLaunchSize, totalSize, inputStride,
            broadcastInputA, broadcastInputB);

    if (totalSize > firstLaunchSize)
    {
        const unsigned int offset = firstLaunchSize;
        DotKernel<<<1, totalSize - firstLaunchSize>>>(
            y, a, b, offset, totalSize - firstLaunchSize,
            totalSize, inputStride, broadcastInputA, broadcastInputB);
    }
}

__host__ void Pow(float* y, const float* x, const float factor,
                  unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        PowKernel<<<blockDim, threadDim>>>(y, x, factor,
                                           firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        PowKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, factor, totalSize - firstLaunchSize);
    }
}

__host__ void log(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        logKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        logKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void log10(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        log10Kernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        log10Kernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}


__host__ void Inverse(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto blockDim = totalSize / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        InverseKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        InverseKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

//! y size should be totalSize/unitSize
__host__ void Mean(float* y, const float* x, unsigned int totalSize,
                   unsigned int unitSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / NUM_LOOPS;

    const auto requiredThreadNum = totalSize / unitSize;
    const auto blockDim = requiredThreadNum / (threadDim * NUM_LOOPS);
    const auto firstLaunchSize = blockDim * threadDim * NUM_LOOPS;

    if (firstLaunchSize > 0)
        MeanKernel<<<blockDim, threadDim>>>(y, x, firstLaunchSize,
                                            unitSize);
    if (requiredThreadNum > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        MeanKernel<<<1, requiredThreadNum - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize, unitSize);
    }
}

} // namespace Sapphire::Compute::Cuda::Dense
