// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/naive/Conv2D.hpp>

namespace Sapphire::Compute::Dense::Naive
{
using namespace TensorUtil;

void Im2Col(TensorData& inputMatrix, TensorData& filter,
            const TensorData& input, int strideCol,
            int strideRow, int rowPadding, int colPadding, int dilationRow,
            int dilationCol, float pad)
{
    const auto inputShape = input.GetShape();
    const auto filterShape = filter.GetShape();
    const auto numChannels = filterShape.At(filterShape.Dim() - 3);
    const auto inputMatrixShape = inputMatrix.GetShape();

    const Shape newFilterShape({ filterShape.Rows(),
                                 filterShape.Size() / filterShape.Rows() });
    filter.TensorShape = newFilterShape;

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

    //! Padded total size of input and inputMatrix per batch
    const auto paddedInputTotalSize =
        (numChannels * inputShape.Rows() * inputShape.Cols() /
         input.PaddedHostColSize) *
        input.PaddedHostColSize +
        (numChannels * inputShape.Rows() * inputShape.Cols()) %
        input.PaddedHostColSize;
    const auto paddedInputMatrixTotalSize =
        ((numChannels * filter.Rows() * filter.Cols() * inputMatrixShape.Cols())
         /
         inputMatrix.PaddedHostColSize) *
        inputMatrix.PaddedHostColSize +
        (numChannels * filter.Rows() * filter.Cols() * inputMatrixShape.Cols())
        %
        inputMatrix.PaddedHostColSize;

    auto* inputMatrixDataHost = inputMatrix.GetMutableDenseHost();
    const auto* inputDataHost = input.GetDenseHost();

    for (int nIdx = 0; nIdx < N; ++nIdx)
    {
        inputDataHost += paddedInputTotalSize * nIdx;
        inputMatrixDataHost += paddedInputMatrixTotalSize * nIdx;
        for (int channelIdx = 0; channelIdx < static_cast<int>(numChannels);
             ++channelIdx)
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

                            const auto inputMatrixColIdx = channelIdx;
                            const auto inputMatrixRowIdx =
                                filterShape.Rows() * filterShape.Cols() *
                                channelIdx +
                                (filterRowIdx * filterShape.Cols() +
                                 filterColIdx);

                            const auto combinedInputMatrixIdx =
                                inputMatrixRowIdx * inputMatrixShape.Cols() +
                                inputMatrixColIdx;
                            const auto combinedInputIdx =
                                (inputRowIdx - rowPadding) * inputShape.Cols() +
                                inputColIdx - colPadding;

                            auto* inputMatrixDataPtr =
                                inputMatrixDataHost +
                                (combinedInputMatrixIdx /
                                 inputMatrix.PaddedHostColSize) *
                                inputMatrix.PaddedHostColSize +
                                combinedInputMatrixIdx %
                                inputMatrix.PaddedHostColSize;

                            const auto* inputDataPtr =
                                inputDataHost +
                                (combinedInputIdx / input.PaddedHostColSize) *
                                input.PaddedHostColSize +
                                combinedInputIdx % input.PaddedHostColSize;

                            if (inputRowIdx > 0 &&
                                inputRowIdx <
                                static_cast<int>(inputShape.Rows()) &&
                                inputColIdx > 0 &&
                                inputColIdx <
                                static_cast<int>(inputShape.Cols()))
                                *inputMatrixDataPtr = *inputDataPtr;
                            else
                                *inputMatrixDataPtr = pad;
                        }
    }
}

void Col2Im(TensorData& input,
            const TensorData& inputMatrix, const TensorData& filter,
            int strideCol,
            int strideRow, int rowPadding, int colPadding, int dilationRow,
            int dilationCol)
{
    const auto inputShape = input.GetShape();
    const auto filterShape = filter.GetShape();
    const auto numChannels = filterShape.At(filterShape.Dim() - 3);

    const Shape newFilterShape(
        { filterShape.Rows(), filterShape.Size() / filterShape.Rows() });

    int N = 1;
    for (unsigned int i = 0; i < inputShape.Dim() - 3; ++i)
        N *= static_cast<int>(inputShape.At(i));

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

    const auto paddedInputTotalSize =
        (numChannels * inputShape.Rows() * inputShape.Cols() /
         input.PaddedHostColSize) *
        input.PaddedHostColSize +
        (numChannels * inputShape.Rows() * inputShape.Cols()) %
        input.PaddedHostColSize;
    const auto paddedInputMatrixTotalSize =
        ((numChannels * filter.Rows() * filter.Cols() * outputCols) /
         inputMatrix.PaddedHostColSize) *
        inputMatrix.PaddedHostColSize +
        (numChannels * filter.Rows() * filter.Cols() * outputCols) %
        inputMatrix.PaddedHostColSize;

    const int inputMatrixColSize = outputCols;

    const auto* inputMatrixDataHost = inputMatrix.GetDenseHost();
    auto* inputDataHost = input.GetMutableDenseHost();

    for (int nIdx = 0; nIdx < static_cast<int>(N); ++nIdx)
    {
        inputDataHost += paddedInputTotalSize * nIdx;
        inputMatrixDataHost += paddedInputMatrixTotalSize * nIdx;
        for (int channelIdx = 0; channelIdx < static_cast<int>(numChannels);
             ++channelIdx)
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

                            const auto inputMatrixColIdx = channelIdx;
                            const auto inputMatrixRowIdx =
                                filterShape.Rows() * filterShape.Cols() *
                                channelIdx +
                                (filterRowIdx * filterShape.Cols() +
                                 filterColIdx);

                            const auto combinedInputMatrixIdx =
                                inputMatrixRowIdx * inputMatrixColSize +
                                inputMatrixColIdx;
                            const auto combinedInputIdx =
                                (inputRowIdx - rowPadding) * inputShape.Cols() +
                                inputColIdx - colPadding;

                            const auto* inputMatrixDataPtr =
                                inputMatrixDataHost +
                                (combinedInputMatrixIdx /
                                 inputMatrix.PaddedHostColSize) *
                                inputMatrix.PaddedHostColSize +
                                combinedInputMatrixIdx %
                                inputMatrix.PaddedHostColSize;

                            auto* inputDataPtr =
                                inputDataHost +
                                (combinedInputIdx / input.PaddedHostColSize) *
                                input.PaddedHostColSize +
                                combinedInputIdx % input.PaddedHostColSize;

                            if (inputRowIdx > 0 &&
                                inputRowIdx <
                                static_cast<int>(inputShape.Rows()) &&
                                inputColIdx > 0 &&
                                inputColIdx <
                                static_cast<int>(inputShape.Cols()))
                                *inputDataPtr = *inputMatrixDataPtr;
                        }
    }
}


} // namespace Sapphire::Comptue
