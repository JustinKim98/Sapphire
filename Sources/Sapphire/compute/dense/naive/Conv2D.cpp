// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cassert>
#include <Sapphire/compute/dense/naive/Conv2D.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/IndexingOps.hpp>

#include "Sapphire/compute/Initialize.hpp"

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
                                *inputDataPtr += *inputMatrixDataPtr;
                        }
        }
    }
}


void Conv2D(TensorData& y, const TensorData& x, const TensorData& filter,
            int strideRow, int strideCol, int rowPadding, int colPadding,
            int dilationRow, int dilationCol, CudaDevice device)
{
    assert(y.Mode() == x.Mode());
    assert(y.Mode() == filter.Mode());
    const auto filterShape = filter.GetShape();
    const auto inputShape = x.GetShape();
    const auto yShape = y.GetShape();
    const auto yRows = y.GetShape().Rows();
    const auto yCols = y.GetShape().Cols();
    const auto N = y.GetBatchSize(3);
    const auto yChannels = yShape.At(yShape.Dim() - 3);

    const auto rXRows = filterShape.At(filterShape.Dim() - 3) *
                        filterShape.Rows() * filterShape.Cols();
    const auto rXCols = static_cast<unsigned int>(yRows * yCols);

    const Shape rXShape({ N, rXRows, rXCols });
    const Shape rFilterShape({ yChannels, filterShape.Size() / yChannels });
    const Shape rYShape({ N, yChannels, yRows * yCols });

    TensorData rX(rXShape, Type::Dense, device);
    TensorData rFilter = filter;
    TensorData rY = y;

    Im2Col(rX, filter, x, strideRow, strideCol, rowPadding, colPadding,
           dilationRow, dilationCol, 0);
    Reshape(rFilter, rFilterShape);
    Reshape(rY, rYShape);

    Gemm(rY, rFilter, rX, rY);

    Reshape(rFilter, filterShape);
    Reshape(rY, yShape);
}

void Conv2DBackward(TensorData& dx, TensorData& dFilter, const TensorData& dy,
                    const TensorData& x, const TensorData& filter,
                    int strideRow,
                    int strideCol, int rowPadding, int colPadding,
                    int dilationRow, int dilationCol, CudaDevice device)
{
    assert(dx.Mode() == DeviceType::Host);
    assert(dFilter.Mode() == DeviceType::Host);
    assert(dy.Mode() == DeviceType::Host);
    assert(x.Mode() == DeviceType::Host);
    assert(filter.Mode() == DeviceType::Host);

    const auto dXShape = dx.GetShape();
    const auto dFilterShape = dFilter.GetShape();
    const auto dYShape = dy.GetShape();
    const auto dyRows = dy.Rows();
    const auto dyCols = dy.Cols();
    const auto N = dy.GetBatchSize(3);
    const auto dyChannels = dYShape.At(dYShape.Dim() - 3);

    const auto drXRows = dFilterShape.At(dFilterShape.Dim() - 3) *
                         dFilterShape.Rows() * dFilterShape.Cols();
    const auto drXCols = static_cast<unsigned int>(dyRows * dyCols);

    const Shape rXShape({ N, drXRows, drXCols });
    const Shape rFilterShape({ dyChannels, dFilterShape.Size() / dyChannels });
    const Shape drYShape({ N, dyChannels, dyRows * dyCols });

    TensorData rX(rXShape, Type::Dense, device);
    TensorData rXT(rXShape.GetTranspose(), Type::Dense, device);
    TensorData drX(rXShape, Type::Dense, device);
    TensorData rFilter = filter;
    TensorData rFilterT(rFilterShape.GetTranspose(), Type::Dense, device);
    TensorData drFilter = dFilter;
    TensorData drY = dy;

    rX.SetMode(DeviceType::Host);
    rXT.SetMode(DeviceType::Host);
    drX.SetMode(DeviceType::Host);
    rFilter.SetMode(DeviceType::Host);
    rFilterT.SetMode(DeviceType::Host);
    drFilter.SetMode(DeviceType::Host);
    drY.SetMode(DeviceType::Host);

    Im2Col(rX, filter, x, strideRow, strideCol, rowPadding, colPadding,
           dilationRow, dilationCol, 0);
    Reshape(rFilter, rFilterShape);
    Reshape(drFilter, rFilterShape);
    Reshape(drY, drYShape);

    Transpose(rFilterT, rFilter);
    Gemm(drX, rFilterT, drY, drX);
    Transpose(rXT, rX);
    Gemm(drFilter, drY, rXT, drFilter);

    Reshape(rFilter, dFilterShape);
    Reshape(drFilter, dFilterShape);
    Reshape(drY, dYShape);
    Col2Im(dx, drX, dFilter, strideCol, strideRow, rowPadding, colPadding,
           dilationRow, dilationCol);
}
} // namespace Sapphire::Comptue
