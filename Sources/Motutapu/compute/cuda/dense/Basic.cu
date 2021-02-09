// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cudnn.h>
#include <Motutapu/compute/cuda/dense/Basic.cuh>
#include <Motutapu/compute/cuda/dense/BasicKernel.cuh>

namespace Motutapu::Compute::Cuda::Dense
{
__host__ void Add(float* output, const float* inputA, const float* inputB,
                  unsigned int totalSize, unsigned int outputBatchSize,
                  unsigned int unitBatchSize)
{
//    const auto numLoops = 8;
//    const auto threadDim = MAX_THREAD_DIM_X / numLoops;
//
//    const auto blockDim = totalSize / (threadDim * numLoops);
//    const auto firstLaunchSize = blockDim * threadDim * numLoops;
//
//    if (firstLaunchSize > 0)
//        AddKernel<<<blockDim, threadDim>>>(output, inputA, inputB,
//                                           firstLaunchSize, inputStride,
//                                           broadcastInputA, broadcastInputB);
//    if (totalSize > firstLaunchSize)
//    {
//        const float* inputAOffset =
//            broadcastInputA ? inputA : inputA + firstLaunchSize;
//        const float* inputBOffset =
//            broadcastInputB ? inputB : inputB + firstLaunchSize;
//        float* outputOffset = output + firstLaunchSize;
//
//        AddKernel<<<1, totalSize - firstLaunchSize>>>(
//            outputOffset, inputAOffset, inputBOffset,
//            totalSize - firstLaunchSize, inputStride, broadcastInputA,
//            broadcastInputB);
//    }
}

__host__ void Sub(float* output, const float* inputA, const float* inputB,
                  unsigned int totalSize, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        SubKernel<<<blockDim, threadDim>>>(output, inputA, inputB,
                                           firstLaunchSize, inputStride,
                                           broadcastInputA, broadcastInputB);
    if (totalSize > firstLaunchSize)
    {
        const float* inputAOffset =
            broadcastInputA ? inputA : inputA + firstLaunchSize;
        const float* inputBOffset =
            broadcastInputB ? inputB : inputB + firstLaunchSize;
        float* outputOffset = output + firstLaunchSize;

        SubKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputAOffset, inputBOffset,
            totalSize - firstLaunchSize, inputStride, broadcastInputA,
            broadcastInputB);
    }
}

__host__ void Dot(float* output, const float* inputA, const float* inputB,
                  unsigned int totalSize, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        SubKernel<<<blockDim, threadDim>>>(output, inputA, inputB,
                                           firstLaunchSize, inputStride,
                                           broadcastInputA, broadcastInputB);
    if (totalSize > firstLaunchSize)
    {
        const float* inputAOffset =
            broadcastInputA ? inputA : inputA + firstLaunchSize;
        const float* inputBOffset =
            broadcastInputB ? inputB : inputB + firstLaunchSize;
        float* outputOffset = output + firstLaunchSize;

        DotKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputAOffset, inputBOffset,
            totalSize - firstLaunchSize, inputStride, broadcastInputA,
            broadcastInputB);
    }
}

__host__ void Scale(float* output, const float* input, const float scaleFactor,
                    unsigned int totalSize, unsigned int inputStride,
                    bool broadcastInput)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        ScaleKernel<<<blockDim, threadDim>>>(output, input, scaleFactor,
                                             firstLaunchSize, inputStride,
                                             broadcastInput);
    if (totalSize > firstLaunchSize)
    {
        const float* inputOffset =
            broadcastInput ? input : input + firstLaunchSize;

        float* outputOffset = output + firstLaunchSize;

        ScaleKernel<<<1, totalSize - firstLaunchSize>>>(
            outputOffset, inputOffset, scaleFactor, totalSize - firstLaunchSize,
            inputStride, broadcastInput);
    }
}

__host__ void Transpose(float* output, const float* input,
                        unsigned int inputNumRows, unsigned int inputNumCols,
                        unsigned int batchSize, bool broadcastInput)
{
    unsigned int blockDimX = (inputNumCols % TILE_DIM == 0)
                                 ? inputNumCols / TILE_DIM
                                 : inputNumCols / TILE_DIM + 1;
    unsigned int blockDimY = (inputNumRows % TILE_DIM == 0)
                                 ? inputNumRows / TILE_DIM
                                 : inputNumRows / TILE_DIM + 1;

    unsigned int blockDimZ = batchSize;
    dim3 blockDim(blockDimX, blockDimY, blockDimZ);
    dim3 threadDim(TILE_DIM, 8);
    TransposeKernel<<<blockDim, threadDim>>>(output, input, inputNumRows,
                                             inputNumCols, broadcastInput);
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
}  // namespace Motutapu::Compute::Cuda::Dense