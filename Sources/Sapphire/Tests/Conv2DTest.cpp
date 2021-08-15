// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/Conv2DTest.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/compute/dense/naive/Conv2D.hpp>
#include <iostream>
#include <random>

#include "doctest.h"

namespace Sapphire::Test
{
void Conv2DTest(bool printForward, bool printBackward)
{
    CudaDevice cuda(0, "cuda0");

    int N = 1;
    int inputHeight = 100;
    int inputWidth = 100;
    int inputChannels = 1;

    int numFilters = 1;
    int filterWidth = 3;
    int filterHeight = 3;
    int strideRow = 2;
    int strideCol = 2;
    int dilationRow = 2;
    int dilationCol = 2;
    int rowPadding = 2;
    int colPadding = 2;

    int outputChannels = numFilters;
    int outputHeight =
        (inputHeight + 2 * rowPadding - dilationRow * (filterHeight - 1) - 1) /
        strideRow +
        1;
    int outputWidth =
        (inputWidth + 2 * colPadding - dilationCol * (filterWidth - 1) - 1) /
        strideCol +
        1;

    Shape xShape({ static_cast<unsigned>(N),
                   static_cast<unsigned>(inputChannels),
                   static_cast<unsigned>(inputHeight),
                   static_cast<unsigned>(inputWidth) });
    Shape filterShape({ static_cast<unsigned>(numFilters),
                        static_cast<unsigned>(inputChannels),
                        static_cast<unsigned>(filterHeight),
                        static_cast<unsigned>(filterWidth) });
    Shape yShape({ static_cast<unsigned>(N),
                   static_cast<unsigned>(outputChannels),
                   static_cast<unsigned>(outputHeight),
                   static_cast<unsigned>(outputWidth) });

    TensorUtil::TensorData x(xShape, Type::Dense, cuda);
    TensorUtil::TensorData dx(xShape, Type::Dense, cuda);
    TensorUtil::TensorData filter(filterShape, Type::Dense, cuda);
    TensorUtil::TensorData dFilter(filterShape, Type::Dense, cuda);
    TensorUtil::TensorData y(yShape, Type::Dense, cuda);
    TensorUtil::TensorData dy(yShape, Type::Dense, cuda);

    x.SetMode(DeviceType::Cuda);
    dx.SetMode(DeviceType::Cuda);
    filter.SetMode(DeviceType::Cuda);
    dFilter.SetMode(DeviceType::Cuda);
    y.SetMode(DeviceType::Cuda);
    dy.SetMode(DeviceType::Cuda);

    Compute::Initialize::Ones(x);
    Compute::Initialize::Zeros(dx);
    Compute::Initialize::Ones(filter);
    Compute::Initialize::Ones(dFilter);
    Compute::Initialize::Zeros(y);
    Compute::Initialize::Ones(dy);

    Compute::Conv2DForward(y, x, filter, strideRow, strideCol, dilationRow,
                           dilationCol, rowPadding, colPadding);

    y.ToHost();
    y.SetMode(DeviceType::Host);

    const auto* forwardOutput = y.GetDenseHost();
    for (unsigned i = 0; i < y.DenseTotalLengthHost; ++i)
        if (printForward)
            std::cout << "forwardData [" << i << "] : " << forwardOutput[i] <<
                std::endl;

    y.SetMode(DeviceType::Cuda);

    Compute::Conv2DBackward(dx, dFilter, dy, x, filter, strideRow, strideCol,
                            dilationRow, dilationCol, rowPadding, colPadding);

    dx.ToHost();

    const auto* backwardOutput = dx.GetDenseHost();
    for (unsigned i = 0; i < dy.DenseTotalLengthHost; ++i)
        if (printBackward)
            std::cout << "backwardData[" << i << "]: " << backwardOutput[i]
                << std::endl;
}

void MaxPool2DTest(bool printForward, bool printBackward)
{
    CudaDevice cuda(0, "cuda0");

    int N = 1;
    int inputHeight = 100;
    int inputWidth = 100;
    int inputChannels = 1;

    int windowRows = 4;
    int windowCols = 4;
    int strideRow = 2;
    int strideCol = 2;
    int rowPadding = 2;
    int colPadding = 2;

    int outputChannels = inputChannels;
    int outputHeight =
        (inputHeight + 2 * rowPadding - (windowCols - 1) - 1) /
        strideRow +
        1;
    int outputWidth =
        (inputWidth + 2 * colPadding - (windowRows - 1) - 1) /
        strideCol +
        1;

    Shape xShape({ static_cast<unsigned>(N),
                   static_cast<unsigned>(inputChannels),
                   static_cast<unsigned>(inputHeight),
                   static_cast<unsigned>(inputWidth) });
    Shape yShape({ static_cast<unsigned>(N),
                   static_cast<unsigned>(outputChannels),
                   static_cast<unsigned>(outputHeight),
                   static_cast<unsigned>(outputWidth) });

    TensorUtil::TensorData x(xShape, Type::Dense, cuda);
    TensorUtil::TensorData dx(xShape, Type::Dense, cuda);
    TensorUtil::TensorData y(yShape, Type::Dense, cuda);
    TensorUtil::TensorData dy(yShape, Type::Dense, cuda);

    x.SetMode(DeviceType::Cuda);
    dx.SetMode(DeviceType::Cuda);
    y.SetMode(DeviceType::Cuda);
    dy.SetMode(DeviceType::Cuda);

    Compute::Initialize::Ones(x);
    Compute::Initialize::Zeros(dx);
    Compute::Initialize::Zeros(y);
    Compute::Initialize::Ones(dy);

    Compute::MaxPool2DForward(y, x, windowRows, windowCols, strideRow,
                              strideCol,
                              rowPadding, colPadding);

    y.ToHost();
    y.SetMode(DeviceType::Host);

    const auto* forwardOutput = y.GetDenseHost();
    for (unsigned i = 0; i < y.DenseTotalLengthHost; ++i)
        if (printForward)
            std::cout << "forwardData [" << i << "] : " << forwardOutput[i]
                << std::endl;

    y.SetMode(DeviceType::Cuda);

    Compute::MaxPool2DBackward(dx, dy, x, y, windowRows, windowCols, strideRow,
                               strideCol, rowPadding, colPadding);

    dx.ToHost();

    const auto* backwardOutput = dx.GetDenseHost();
    for (unsigned i = 0; i < dy.DenseTotalLengthHost; ++i)
        if (printBackward)
            std::cout << "backwardData[" << i << "]: " << backwardOutput[i]
                << std::endl;
}

void AvgPool2DTest(bool printForward, bool printBackward)
{
    CudaDevice cuda(0, "cuda0");

    int N = 1;
    int inputHeight = 100;
    int inputWidth = 100;
    int inputChannels = 1;

    int windowRows = 4;
    int windowCols = 4;
    int strideRow = 2;
    int strideCol = 2;
    int rowPadding = 2;
    int colPadding = 2;

    int outputChannels = inputChannels;
    int outputHeight =
        (inputHeight + 2 * rowPadding - (windowCols - 1) - 1) / strideRow + 1;
    int outputWidth =
        (inputWidth + 2 * colPadding - (windowRows - 1) - 1) / strideCol + 1;

    Shape xShape({ static_cast<unsigned>(N),
                   static_cast<unsigned>(inputChannels),
                   static_cast<unsigned>(inputHeight),
                   static_cast<unsigned>(inputWidth) });
    Shape yShape({ static_cast<unsigned>(N),
                   static_cast<unsigned>(outputChannels),
                   static_cast<unsigned>(outputHeight),
                   static_cast<unsigned>(outputWidth) });

    TensorUtil::TensorData x(xShape, Type::Dense, cuda);
    TensorUtil::TensorData dx(xShape, Type::Dense, cuda);
    TensorUtil::TensorData y(yShape, Type::Dense, cuda);
    TensorUtil::TensorData dy(yShape, Type::Dense, cuda);

    x.SetMode(DeviceType::Cuda);
    dx.SetMode(DeviceType::Cuda);
    y.SetMode(DeviceType::Cuda);
    dy.SetMode(DeviceType::Cuda);

    Compute::Initialize::Ones(x);
    Compute::Initialize::Zeros(dx);
    Compute::Initialize::Zeros(y);
    Compute::Initialize::Ones(dy);

    Compute::AvgPool2DForward(y, x, windowRows, windowCols, strideRow,
                              strideCol, rowPadding, colPadding);

    y.ToHost();
    y.SetMode(DeviceType::Host);

    const auto* forwardOutput = y.GetDenseHost();
    for (unsigned i = 0; i < y.DenseTotalLengthHost; ++i)
        if (printForward)
            std::cout << "forwardData [" << i << "] : " << forwardOutput[i]
                << std::endl;

    y.SetMode(DeviceType::Cuda);

    Compute::AvgPool2DBackward(dx, dy, x, y, windowRows, windowCols, strideRow,
                               strideCol, rowPadding, colPadding);

    dx.ToHost();

    const auto* backwardOutput = dx.GetDenseHost();
    for (unsigned i = 0; i < dy.DenseTotalLengthHost; ++i)
        if (printBackward)
            std::cout << "backwardData[" << i << "]: " << backwardOutput[i]
                << std::endl;
}

void HostIm2ColTest(bool print)
{
    const int N = 2;
    const int numFilters = 2;
    const int numInputChannels = 3;
    const int InputRows = 10;
    const int InputCols = 10;
    const int filterRows = 3;
    const int filterCols = 2;
    const int rowPadding = 1;
    const int colPadding = 1;
    const int dilationRow = 2;
    const int dilationCol = 1;
    const int strideRow = 2;
    const int strideCol = 3;

    const Shape inputShape({ N, numInputChannels, InputRows, InputCols });
    const Shape filterShape({ numFilters, numInputChannels, filterRows,
                              filterCols });

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

    const Shape outputShape({ N, numFilters,
                              static_cast<unsigned int>(outputRows),
                              static_cast<unsigned int>(outputCols) });

    const auto inputMatrixRows =
        filterShape.At(filterShape.Dim() - 3) * filterShape.Rows() * filterShape
        .
        Cols();
    const auto inputMatrixCols =
        static_cast<unsigned int>(outputRows * outputCols);

    const Shape inputMatrixShape({ N, inputMatrixRows, inputMatrixCols });

    CudaDevice device(0, "cuda0");
    TensorUtil::TensorData inputData(inputShape, Type::Dense, device);
    TensorUtil::TensorData filterData(filterShape, Type::Dense, device);
    TensorUtil::TensorData inputMatrixData(inputMatrixShape, Type::Dense,
                                           device);

    TensorUtil::TensorData
        reConvertedInputData(inputShape, Type::Dense, device);

    int count = 0;
    for (std::size_t ii = 0; ii < inputData.GetBatchSize(1); ++ii)
        for (unsigned int i = 0; i < inputData.Cols(); ++i)
            inputData
                .GetMutableDenseHost()[ii * inputData.PaddedHostColSize + i] =
                static_cast<float>((count++) % 9);

    Compute::Dense::Naive::Im2Col(inputMatrixData, filterData, inputData,
                                  strideRow, strideCol, rowPadding,
                                  colPadding, dilationRow, dilationCol, 0);

    if (print)
        for (std::size_t ii = 0; ii < inputMatrixData.GetBatchSize(1); ++ii)
            for (unsigned int i = 0; i < inputMatrixData.Cols(); ++i)
                std::cout << "Im2Col[" << ii * inputMatrixData.Cols() + i
                    << "]: " << inputMatrixData.GetDenseHost()[
                        ii * inputMatrixData.PaddedHostColSize + i]
                    << std::endl;

    Compute::Dense::Naive::Col2Im(inputData, inputMatrixData, filterData,
                                  strideCol, strideRow, rowPadding, colPadding,
                                  dilationRow, dilationCol);

    const Shape newFilterShape(
        { filterShape.Rows(), filterShape.Size() / filterShape.Rows() });
    filterData.TensorShape = newFilterShape;

    if (print)
        for (std::size_t ii = 0; ii < inputData.GetBatchSize(1); ++ii)
            for (unsigned int i = 0; i < inputData.Cols(); ++i)
                std::cout << "Col2Im[" << ii * inputData.Cols() + i
                    << "] = " <<
                    inputData.GetMutableDenseHost()
                    [ii * inputData.PaddedHostColSize + i]
                    << std::endl;
}

void HostConv2DForwardTest(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-10, 10);

    const int N = 2;
    const int numFilters = 5;
    const int numInputChannels = 1;
    const int InputRows = 10;
    const int InputCols = 18;
    const int filterRows = 2;
    const int filterCols = 3;
    const int rowPadding = 2;
    const int colPadding = 2;
    const int dilationRow = 2;
    const int dilationCol = 1;
    const int strideRow = 2;
    const int strideCol = 3;

    const Shape inputShape({ N, numInputChannels, InputRows, InputCols });
    const Shape filterShape(
        { numFilters, numInputChannels, filterRows, filterCols });

    const int yRows =
        (static_cast<int>(inputShape.Rows()) + 2 * rowPadding -
         dilationRow * (filterShape.Rows() - 1) - 1) /
        strideRow +
        1;
    const int yCols =
        (static_cast<int>(inputShape.Cols()) + 2 * colPadding -
         dilationCol * (filterShape.Cols() - 1) - 1) /
        strideCol +
        1;

    const Shape yShape({ N, numFilters, static_cast<unsigned int>(yRows),
                         static_cast<unsigned int>(yCols) });

    CudaDevice device(0, "cuda0");
    TensorUtil::TensorData x(inputShape, Type::Dense, device);
    TensorUtil::TensorData filter(filterShape, Type::Dense, device);
    TensorUtil::TensorData y(yShape, Type::Dense, device);
    Compute::Initialize::Zeros(y);

    int count = 0;
    for (std::size_t ii = 0; ii < x.GetBatchSize(1); ++ii)
        for (unsigned int i = 0; i < x.Cols(); ++i)
            x.GetMutableDenseHost()[ii * x.PaddedHostColSize + i] = static_cast<
                float>(distrib(gen));

    count = 0;
    for (std::size_t ii = 0; ii < filter.GetBatchSize(1); ++ii)
        for (unsigned int i = 0; i < filter.Cols(); ++i)
            filter.GetMutableDenseHost()[ii * filter.PaddedHostColSize + i] =
                static_cast<float>(distrib(gen));

    Compute::Dense::Naive::Conv2D(y, x, filter, strideRow, strideCol,
                                  rowPadding, colPadding, dilationRow,
                                  dilationCol, device);

    auto* hostTemp = new float[y.GetShape().Size()];

    for (std::size_t ii = 0; ii < y.GetBatchSize(1); ++ii)
        for (unsigned int i = 0; i < y.Cols(); ++i)
            hostTemp[ii * y.Cols() + i] =
                y.GetMutableDenseHost()[ii * y.PaddedHostColSize + i];

    x.ToCuda();
    filter.ToCuda();
    x.SetMode(DeviceType::Cuda);
    filter.SetMode(DeviceType::Cuda);
    y.SetMode(DeviceType::Cuda);

    Compute::Conv2DForward(y, x, filter, strideRow, strideCol, dilationRow,
                           dilationCol,
                           rowPadding, colPadding);

    y.ToHost();

    for (std::size_t ii = 0; ii < y.GetBatchSize(1); ++ii)
        for (unsigned int i = 0; i < y.Cols(); ++i)
        {
            if (!std::isnan(hostTemp[ii * y.Cols() + i]) &&
                !std::isnan(
                    y.GetMutableDenseHost()[ii * y.PaddedHostColSize + i]))
                CHECK(std::abs(
                    hostTemp[ii * y.Cols() + i] -
                    y.GetMutableDenseHost()[ii * y.PaddedHostColSize + i]) <
                0.01f);
            if (print)
                std::cout
                    << "host[" << ii * y.Cols() + i
                    << "] = " << hostTemp[ii * y.Cols() + i] << "  cuda["
                    << ii * y.Cols() + i << "] = "
                    << y.GetMutableDenseHost()[ii * y.PaddedHostColSize + i]
                    << std::endl;
        }

    delete[] hostTemp;
}
}
