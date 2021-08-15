// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/naive/Conv2D.hpp>
#include <Sapphire/compute/BasicOps.hpp>

namespace Sapphire::Compute::Dense::Naive
{
using namespace TensorUtil;

void Im2Col(TensorData& inputMatrix, const TensorData& filter,
            const TensorData& input, int strideRow, int strideCol,
            int rowPadding, int colPadding, int dilationRow, int dilationCol,
            float pad)
{
    const auto inputShape = input.GetShape();
    const auto filterShape = filter.GetShape();
    const auto numChannels = filterShape.At(filterShape.Dim() - 3);
    const auto inputMatrixShape = inputMatrix.GetShape();

    int N = 1;
    for (unsigned int i = 0; i < inputShape.Dim() - 3; ++i)
        N *= static_cast<int>(inputShape.At(i));

    //! Size of the output after the full convolution
    const int outputRows =
        (static_cast<int>(inputShape.Rows()) + 2 * rowPadding -
         dilationRow * (filterShape.Rows() - 1) - 1) /
        strideRow +
        1;
    const int outputCols =
        (static_cast<int>(inputShape.Cols()) + 2 * colPadding -
         dilationCol * (filterShape.Cols() - 1) - 1) /
        strideCol +
        1;

    const auto InputSizePerBatch =
        numChannels * inputShape.Rows() * inputShape.Cols();
    const auto InputMatrixSizePerBatch =
        inputMatrixShape.Rows() * inputMatrixShape.Cols();

    //! Padded total size of input and inputMatrix per batch
    const auto paddedInputTotalSize =
        (InputSizePerBatch / input.Cols()) * input.PaddedHostColSize;
    const auto paddedInputMatrixTotalSize =
        (InputMatrixSizePerBatch / inputMatrix.Cols()) *
        inputMatrix.PaddedHostColSize;

    auto* inputMatrixDataHost = inputMatrix.GetMutableDenseHost();
    const auto* inputDataHost = input.GetDenseHost();

    for (int nIdx = 0; nIdx < N; ++nIdx)
    {
        inputDataHost += paddedInputTotalSize * nIdx;
        inputMatrixDataHost += paddedInputMatrixTotalSize * nIdx;
        for (int channelIdx = 0; channelIdx < static_cast<int>(numChannels);
             ++channelIdx)
        {
            for (int outputRowIdx = 0; outputRowIdx < outputRows;
                 outputRowIdx += 1)
                for (int outputColIdx = 0; outputColIdx < outputCols;
                     outputColIdx += 1)
                    for (int filterRowIdx = 0;
                         filterRowIdx < static_cast<int>(filterShape.Rows());
                         ++filterRowIdx)
                        for (int filterColIdx = 0;
                             filterColIdx <
                             static_cast<int>(filterShape.Cols());
                             ++filterColIdx)
                        {
                            const auto windowRowIdx = outputRowIdx * strideRow;
                            const auto windowColIdx = outputColIdx * strideCol;

                            const auto rowIdx =
                                windowRowIdx + filterRowIdx * dilationRow;
                            const auto colIdx =
                                windowColIdx + filterColIdx * dilationCol;

                            const auto inputRowIdx = rowIdx - rowPadding;
                            const auto inputColIdx = colIdx - colPadding;

                            const auto inputMatrixRowIdx =
                                filterShape.Rows() * filterShape.Cols() *
                                channelIdx +
                                filterShape.Rows() * filterShape.Cols() -
                                (filterRowIdx * filterShape.Cols() +
                                 filterColIdx) -
                                1;
                            const auto inputMatrixColIdx =
                                outputRowIdx * outputCols + outputColIdx;

                            const auto combinedInputMatrixIdx =
                                inputMatrixRowIdx * inputMatrixShape.Cols() +
                                inputMatrixColIdx;
                            const auto combinedInputIdx =
                                inputShape.Rows() * inputShape.Cols() *
                                channelIdx +
                                inputRowIdx * inputShape.Cols() + inputColIdx;

                            auto* inputMatrixDataPtr =
                                inputMatrixDataHost +
                                (combinedInputMatrixIdx / inputMatrix.Cols()) *
                                inputMatrix.PaddedHostColSize +
                                combinedInputMatrixIdx % inputMatrix.Cols();

                            const auto* inputDataPtr =
                                inputDataHost +
                                (combinedInputIdx / input.Cols()) *
                                input.PaddedHostColSize +
                                combinedInputIdx % input.Cols();

                            if (inputRowIdx >= 0 &&
                                inputRowIdx <
                                static_cast<int>(inputShape.Rows()) &&
                                inputColIdx >= 0 &&
                                inputColIdx <
                                static_cast<int>(inputShape.Cols()))
                                *inputMatrixDataPtr = *inputDataPtr;
                            else
                                *inputMatrixDataPtr = pad;
                        }
        }
    }
}

void Col2Im(TensorData& input, const TensorData& inputMatrix,
            const TensorData& filter, int strideCol, int strideRow,
            int rowPadding, int colPadding, int dilationRow, int dilationCol)
{
    const auto inputShape = input.GetShape();
    const auto filterShape = filter.GetShape();
    const auto numChannels = filterShape.At(filterShape.Dim() - 3);
    const auto inputMatrixShape = inputMatrix.GetShape();

    const Shape newFilterShape(
        { filterShape.Rows(), filterShape.Size() / filterShape.Rows() });

    int N = 1;
    for (unsigned int i = 0; i < inputShape.Dim() - 3; ++i)
        N *= static_cast<int>(inputShape.At(i));

    //! Size of the output after the full convolution
    const int outputRows =
        (static_cast<int>(inputShape.Rows()) + 2 * rowPadding -
         dilationRow * (filterShape.Rows() - 1) - 1) /
        strideRow +
        1;
    const int outputCols =
        (static_cast<int>(inputShape.Cols()) + 2 * colPadding -
         dilationCol * (filterShape.Cols() - 1) - 1) /
        strideCol +
        1;

    const auto InputSizePerBatch =
        numChannels * inputShape.Rows() * inputShape.Cols();
    const auto InputMatrixSizePerBatch =
        inputMatrixShape.Rows() * inputMatrixShape.Cols();

    //! Padded total size of input and inputMatrix per batch
    const auto paddedInputTotalSize =
        (InputSizePerBatch / input.Cols()) * input.PaddedHostColSize;
    const auto paddedInputMatrixTotalSize =
        (InputMatrixSizePerBatch / inputMatrix.Cols()) *
        inputMatrix.PaddedHostColSize;

    const auto* inputMatrixDataHost = inputMatrix.GetDenseHost();
    auto* inputDataHost = input.GetMutableDenseHost();

    for (int nIdx = 0; nIdx < N; ++nIdx)
    {
        inputDataHost += paddedInputTotalSize * nIdx;
        inputMatrixDataHost += paddedInputMatrixTotalSize * nIdx;
        for (int channelIdx = 0; channelIdx < static_cast<int>(numChannels);
             ++channelIdx)
        {
            for (int outputRowIdx = 0; outputRowIdx < outputRows;
                 outputRowIdx += 1)
                for (int outputColIdx = 0; outputColIdx < outputCols;
                     outputColIdx += 1)
                    for (int filterRowIdx = 0;
                         filterRowIdx < static_cast<int>(filterShape.Rows());
                         ++filterRowIdx)
                        for (int filterColIdx = 0;
                             filterColIdx <
                             static_cast<int>(filterShape.Cols());
                             ++filterColIdx)
                        {
                            const auto windowRowIdx = outputRowIdx * strideRow;
                            const auto windowColIdx = outputColIdx * strideCol;

                            const auto rowIdx =
                                windowRowIdx + filterRowIdx * dilationRow;
                            const auto colIdx =
                                windowColIdx + filterColIdx * dilationCol;

                            const auto inputRowIdx = rowIdx - rowPadding;
                            const auto inputColIdx = colIdx - colPadding;

                            const auto inputMatrixRowIdx =
                                filterShape.Rows() * filterShape.Cols() *
                                channelIdx +
                                filterShape.Rows() * filterShape.Cols() -
                                (filterRowIdx * filterShape.Cols() +
                                 filterColIdx) -
                                1;
                            const auto inputMatrixColIdx =
                                outputRowIdx * outputCols + outputColIdx;

                            const auto combinedInputMatrixIdx =
                                inputMatrixRowIdx * inputMatrixShape.Cols() +
                                inputMatrixColIdx;
                            const auto combinedInputIdx =
                                inputShape.Rows() * inputShape.Cols() *
                                channelIdx +
                                inputRowIdx * inputShape.Cols() + inputColIdx;

                            const auto* inputMatrixDataPtr =
                                inputMatrixDataHost +
                                (combinedInputMatrixIdx / inputMatrix.Cols()) *
                                inputMatrix.PaddedHostColSize +
                                combinedInputMatrixIdx % inputMatrix.Cols();

                            auto* inputDataPtr =
                                inputDataHost +
                                (combinedInputIdx / input.Cols()) *
                                input.PaddedHostColSize +
                                combinedInputIdx % input.Cols();

                            if (inputRowIdx >= 0 &&
                                inputRowIdx <
                                static_cast<int>(inputShape.Rows()) &&
                                inputColIdx >= 0 &&
                                inputColIdx <
                                static_cast<int>(inputShape.Cols()))
                                *inputDataPtr = *inputMatrixDataPtr;
                        }
        }
    }
}

void ReshapeFilter(TensorData& filter)
{
    const Shape filterShape = filter.GetShape();
    const Shape newFilterShape(
        { filterShape.Rows(), filterShape.Size() / filterShape.Rows() });

    const auto padUnitSize = static_cast<unsigned long>(32 / sizeof(float));

    const auto paddedColumnSize =
        newFilterShape.Cols() % padUnitSize == 0
            ? newFilterShape.Cols()
            : newFilterShape.Cols() / padUnitSize * padUnitSize + padUnitSize;

    auto* tempData = new float[filterShape.Size()];
    for (std::size_t ii = 0; ii < filter.GetBatchSize(1); ++ii)
        for (unsigned int i = 0; i < filter.Cols(); ++i)
        {
            auto data =
                filter.GetDenseHost()[ii * filter.PaddedHostColSize + i];
            tempData[ii * filterShape.Cols() + i] = data;
        }

    filter.PaddedHostColSize = paddedColumnSize;
    filter.TensorShape = newFilterShape;

    for (std::size_t ii = 0; ii < filter.GetBatchSize(1); ++ii)
        for (unsigned int i = 0; i < filter.Cols(); ++i)
            filter.GetMutableDenseHost()[
                    ii * filter.PaddedHostColSize + i] =
                tempData[ii * newFilterShape
                         .Cols() + i];

    delete[] tempData;
}

void ReshapeOutput(TensorData& output)
{
    const Shape outputShape = output.GetShape();
    std::vector<unsigned> newShapeVector(outputShape.Dim() - 1);

    for (unsigned i = 0; i < outputShape.Dim() - 1; ++i)
        newShapeVector[i] = outputShape.At(i);

    newShapeVector[newShapeVector.size() - 1] *= outputShape.Cols();
    const auto newOutputShape = Shape(newShapeVector);

    const auto padUnitSize = static_cast<unsigned long>(32 / sizeof(float));

    const auto paddedColumnSize =
        newOutputShape.Cols() % padUnitSize == 0
            ? newOutputShape.Cols()
            : newOutputShape.Cols() / padUnitSize * padUnitSize + padUnitSize;

    output.PaddedHostColSize = paddedColumnSize;

    output.TensorShape = newOutputShape;
}

void Conv2D(TensorData& y, const TensorData& input, const TensorData& filter,
            int strideRow, int strideCol, int rowPadding, int colPadding,
            int dilationRow, int dilationCol, CudaDevice device)
{
    const auto filterShape = filter.GetShape();
    const auto yRows = y.GetShape().Rows();
    const auto yCols = y.GetShape().Cols();
    const unsigned int N = y.GetBatchSize(3);

    const auto inputMatrixRows = filterShape.At(filterShape.Dim() - 3) *
                                 filterShape.Rows() * filterShape.Cols();
    const auto inputMatrixCols =
        static_cast<unsigned int>(yRows * yCols);

    const Shape inputMatrixShape({ N, inputMatrixRows, inputMatrixCols });
    TensorUtil::TensorData inputMatrix(inputMatrixShape, Type::Dense,
                                       device);
    TensorUtil::TensorData reshapedFilter = filter;
    TensorUtil::TensorData yMatrix = y;

    ReshapeFilter(reshapedFilter);
    ReshapeOutput(yMatrix);

    Im2Col(inputMatrix, filter, input, strideRow, strideCol, rowPadding,
           colPadding, dilationRow, dilationCol, 0);

    Compute::Gemm(yMatrix, reshapedFilter, inputMatrix, yMatrix);
}
} // namespace Sapphire::Comptue
