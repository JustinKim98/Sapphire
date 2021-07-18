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
const unsigned int numLoops = 8;

__host__ void Add(unsigned int totalSize, float* y, const float* a,
                  const float* b, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

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
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

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
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

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
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void cos(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void sin(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void tan(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void cosh(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void sinh(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void tanh(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void log(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void ReLU(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void ReLUDerivative(float* y, const float* x,
                             unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        ReLUDerivativeKernel<<<blockDim, threadDim>>>(y, x,
            firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        ReLUDerivativeKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, totalSize - firstLaunchSize);
    }
}

__host__ void LeakyReLU(float* y, const float* x, const float a,
                        unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        LeakyReLUKernel<<<blockDim, threadDim>>>(y, x, a,
                                                 firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        LeakyReLUKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, a, totalSize - firstLaunchSize);
    }
}

__host__ void LeakyReLUBackward(float* y, const float* x,
                                const float a, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        LeakyReLUDerivativeKernel<<<blockDim, threadDim>>>(y, x, a,
            firstLaunchSize);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset = x + firstLaunchSize;
        float* outputOffset = y + firstLaunchSize;

        LeakyReLUDerivativeKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, a, totalSize - firstLaunchSize);
    }
}

__host__ void Inverse(float* y, const float* x, unsigned int totalSize)
{
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto requiredThreadNum = totalSize / unitSize;
    const auto blockDim = requiredThreadNum / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

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

__host__ void Softmax(float* y, const float* x, unsigned int totalSize,
                      unsigned int unitSize)
{
    const auto blockDim = (unitSize > 512) ? 512 : unitSize;
    const auto gridDim = (totalSize % blockDim == 0)
                             ? totalSize / blockDim
                             : totalSize / blockDim + 1;
    SoftmaxKernel<<<gridDim, blockDim>>>(y, x, totalSize, unitSize);
}

__host__ void SoftmaxBack(float* dx, const float* dy, const float* x,
                          unsigned int totalSize, unsigned int unitSize,
                          unsigned int padSize)
{
    auto blockDim = (unitSize > 512) ? 512 : unitSize;
    const auto gridDim = (totalSize % blockDim == 0)
                             ? totalSize / blockDim
                             : totalSize / blockDim + 1;
    SoftmaxBackKernel<<<gridDim, blockDim>>>(dx, dy, x, totalSize, unitSize);
}

//__global__ void ConvInputToFeatureMatrix(
//    float* out, float* input, unsigned int inputChannels,
//    unsigned int inputRows, unsigned int inputColumns,
//    unsigned int inputPaddedRows, unsigned int inputPaddedColumns,
//    unsigned int outputPaddedRows, unsigned int outputPaddedColumns,
//    unsigned int filterRows, unsigned int filterCols, unsigned int padSizeRow,
//    unsigned int padSizeCol, unsigned int strideRow, unsigned int strideCol,
//    unsigned int dilationRow, unsigned int dilationCol)
//{
//    const int threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    const int inputMatrixSize = inputPaddedRows * inputPaddedColumns;
//    const int convPerRow =
//        (inputRows - filterRows + 1 + padSizeRow * 2) / strideRow;
//    const int convPerCol =
//        (inputCols - filterCols + 1 + padSizeCol * 2) / strideCol;
//
//    const int channelIdx = threadIdx / convPerRow * convPerCol;
//    const int convRowIdx = (threadIdx % convPerRow * convPerCol) / convPerRow;
//    const int convColIdx = (threadIdx % convPerRow * convPerCol) / convPerCol;
//
//    float* inputStartOffset = input + inputMatrixSize * channelIdx +
//                              inputPaddedColumns * strideRow * convRowIdx +
//                              strideCol * convColIdx;
//
//    float* outputStartOffset =
//        output + outputPaddedColumns * (convPerRow * convRowIdx + convColIdx)
//        + filterRows * filterCols * channelIdx;
//
//    for (int i = 0; i < filterRows; i++)
//        for (j = 0; j < filterCols; j++)
//        {
//            *(inputStartOffset + inputPaddedColumns * i + j)
//        }
//}
} // namespace Sapphire::Compute::Cuda::Dense
