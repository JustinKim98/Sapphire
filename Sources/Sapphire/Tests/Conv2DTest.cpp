// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/Conv2DTest.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/compute/dense/naive/Conv2D.hpp>
#include <Sapphire/util/Shape.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "doctest.h"

namespace Sapphire::Test
{
void Conv2DTest(bool printForward, bool printBackward)
{
    CudaDevice cuda(0, "cuda0");

    int N = 4;
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

    Shape xShape({ N, inputChannels, inputHeight, inputWidth });
    Shape filterShape({ numFilters,
                        inputChannels,
                        filterHeight,
                        filterWidth });
    Shape yShape({ N,
                   outputChannels,
                   outputHeight,
                   outputWidth });

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

    const auto* forwardOutput = y.HostRawPtr();
    for (unsigned i = 0; i < y.HostTotalSize; ++i)
        if (printForward)
            std::cout << "forwardData [" << i << "] : " << forwardOutput[i] <<
                std::endl;

    y.SetMode(DeviceType::Cuda);

    Compute::Conv2DBackward(dx, dFilter, dy, x, filter, strideRow, strideCol,
                            rowPadding, colPadding, dilationRow, dilationCol);

    dx.ToHost();

    const auto* backwardOutput = dx.HostRawPtr();
    for (unsigned i = 0; i < dy.HostTotalSize; ++i)
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

    Shape xShape({ (N),
                   (inputChannels),
                   (inputHeight),
                   (inputWidth) });
    Shape yShape({ (N),
                   (outputChannels),
                   (outputHeight),
                   (outputWidth) });

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

    const auto* forwardOutput = y.HostRawPtr();
    for (unsigned i = 0; i < y.HostTotalSize; ++i)
        if (printForward)
            std::cout << "forwardData [" << i << "] : " << forwardOutput[i]
                << std::endl;

    y.SetMode(DeviceType::Cuda);

    Compute::MaxPool2DBackward(dx, dy, x, y, windowRows, windowCols, strideRow,
                               strideCol, rowPadding, colPadding);

    dx.ToHost();

    const auto* backwardOutput = dx.HostRawPtr();
    for (unsigned i = 0; i < dy.HostTotalSize; ++i)
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

    Shape xShape({ (N),
                   (inputChannels),
                   (inputHeight),
                   (inputWidth) });
    Shape yShape({ (N),
                   (outputChannels),
                   (outputHeight),
                   (outputWidth) });

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

    const auto* forwardOutput = y.HostRawPtr();
    for (unsigned i = 0; i < y.HostTotalSize; ++i)
        if (printForward)
            std::cout << "forwardData [" << i << "] : " << forwardOutput[i]
                << std::endl;

    y.SetMode(DeviceType::Cuda);

    Compute::AvgPool2DBackward(dx, dy, x, y, windowRows, windowCols, strideRow,
                               strideCol, rowPadding, colPadding);

    dx.ToHost();

    const auto* backwardOutput = dx.HostRawPtr();
    for (unsigned i = 0; i < dy.HostTotalSize; ++i)
        if (printBackward)
            std::cout << "backwardData[" << i << "]: " << backwardOutput[i]
                << std::endl;
}

void HostIm2ColTest(bool print)
{
    const int N = 4;
    const int numFilters = 2;
    const int numInputChannels = 3;
    const int InputRows = 3;
    const int InputCols = 3;
    const int filterRows = 2;
    const int filterCols = 2;
    const int rowPadding = 0;
    const int colPadding = 0;
    const int dilationRow = 1;
    const int dilationCol = 1;
    const int strideRow = 1;
    const int strideCol = 1;

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

    const auto inputMatrixRows =
        filterShape.At(filterShape.Dim() - 3) * filterShape.Rows() * filterShape
        .
        Cols();
    const auto inputMatrixCols =
        (outputRows * outputCols);

    const Shape inputMatrixShape({ N, inputMatrixRows, inputMatrixCols });

    CudaDevice device(0, "cuda0");
    TensorUtil::TensorData inputData(inputShape, Type::Dense, device);
    TensorUtil::TensorData filterData(filterShape, Type::Dense, device);
    TensorUtil::TensorData inputMatrixData(inputMatrixShape, Type::Dense,
                                           device);
    TensorUtil::TensorData
        reConvertedInputData(inputShape, Type::Dense, device);

    inputData.SetMode(DeviceType::Host);
    filterData.SetMode(DeviceType::Host);
    inputMatrixData.SetMode(DeviceType::Host);
    reConvertedInputData.SetMode(DeviceType::Host);

    int count = 0;
    for (int i = 0; i < inputData.Size(); ++i)
        inputData.HostMutableRawPtr()[i] = static_cast<float>((count++) % 9);

    Compute::Dense::Naive::Im2Col(inputMatrixData, filterData, inputData,
                                  strideRow, strideCol, rowPadding,
                                  colPadding, dilationRow, dilationCol, 0);

    if (print)
        for (int i = 0; i < inputMatrixData.Size(); ++i)
            std::cout << "Im2Col[" << i << "]: "
                << inputMatrixData.HostRawPtr()[i]
                << std::endl;

    Compute::Initialize::Zeros(inputData);

    Compute::Dense::Naive::Col2Im(inputData, inputMatrixData, filterData,
                                  strideCol, strideRow, rowPadding, colPadding,
                                  dilationRow, dilationCol);

    const Shape newFilterShape(
        { filterShape.Rows(), filterShape.Size() / filterShape.Rows() });
    filterData.Reshape(newFilterShape);

    if (print)
        for (int i = 0; i < inputData.Size(); ++i)
            std::cout << "Col2Im[" << i << "] = "
                << inputData.HostMutableRawPtr()[i]
                << std::endl;
}

void HostConv2DTest(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-10.0f, 10.0f);

    const int N = 2;
    const int numFilters = 5;
    const int numInputChannels = 4;
    const int InputRows = 6;
    const int InputCols = 6;
    const int filterRows = 2;
    const int filterCols = 2;
    const int rowPadding = 2;
    const int colPadding = 2;
    const int dilationRow = 2;
    const int dilationCol = 2;
    const int strideRow = 2;
    const int strideCol = 2;

    const Shape xShape({ N, numInputChannels, InputRows, InputCols });
    const Shape filterShape(
        { numFilters, numInputChannels, filterRows, filterCols });

    const int yRows =
        (static_cast<int>(xShape.Rows()) + 2 * rowPadding -
         dilationRow * (filterShape.Rows() - 1) - 1) /
        strideRow +
        1;
    const int yCols =
        (static_cast<int>(xShape.Cols()) + 2 * colPadding -
         dilationCol * (filterShape.Cols() - 1) - 1) /
        strideCol +
        1;

    const Shape yShape({ N, numFilters, (yRows),
                         (yCols) });

    CudaDevice device(0, "cuda0");
    TensorUtil::TensorData x(xShape, Type::Dense, device);
    TensorUtil::TensorData dx(xShape, Type::Dense, device);
    TensorUtil::TensorData filter(filterShape, Type::Dense, device);
    TensorUtil::TensorData y(yShape, Type::Dense, device);
    TensorUtil::TensorData dFilter(filterShape, Type::Dense, device);
    TensorUtil::TensorData dy(yShape, Type::Dense, device);
    x.SetMode(DeviceType::Host);
    dx.SetMode(DeviceType::Host);
    filter.SetMode(DeviceType::Host);
    y.SetMode(DeviceType::Host);
    dFilter.SetMode(DeviceType::Host);
    dy.SetMode(DeviceType::Host);

    Compute::Initialize::Zeros(y);

    std::vector<float> filterData(filter.Size());
    std::vector<float> xData(x.Size());

    for (auto& data : filterData)
        data = dist(gen);
    for (auto& data : xData)
        data = dist(gen);

    x.SetData(xData);
    filter.SetData(filterData);

    Compute::Dense::Naive::Conv2D(y, x, filter, strideRow, strideCol,
                                  rowPadding, colPadding, dilationRow,
                                  dilationCol, device);

    auto yDataHost = y.GetDataCopy();

    x.ToCuda();
    y.ToCuda();
    filter.ToCuda();

    Compute::Conv2DForward(y, x, filter, strideRow, strideCol, dilationRow,
                           dilationCol,
                           rowPadding, colPadding);

    auto yDataCuda = y.GetDataCopy();

    for (int i = 0; i < y.Size(); ++i)
    {
        if (!std::isnan(yDataHost[i]) &&
            !std::isnan(yDataCuda[i]))
            CHECK(std::abs(
            yDataHost[i] - yDataCuda[i]) < 0.01f);
        if (print)
            std::cout
                << "host[" << i << "] = " << yDataHost[i]
                << "  cuda[" << i << "] = " << yDataCuda[i]
                << std::endl;
    }

    std::vector<float> dyData(y.Size());
    for (auto& data : dyData)
        data = dist(gen);

    dx.ToHost();
    dFilter.ToHost();
    dy.ToHost();
    x.ToHost();
    filter.ToHost();
    Compute::Initialize::Zeros(dx);
    Compute::Initialize::Zeros(dFilter);

    Compute::Dense::Naive::Conv2DBackward(dx, dFilter, dy, x, filter, strideRow,
                                          strideCol, rowPadding, colPadding,
                                          dilationRow, dilationCol, device);

    auto dxDataHost = dx.GetDataCopy();
    auto dFilterDataHost = dFilter.GetDataCopy();

    Compute::Initialize::Zeros(dx);
    Compute::Initialize::Zeros(dFilter);

    x.ToCuda();
    dx.ToCuda();
    filter.ToCuda();
    dFilter.ToCuda();
    dy.ToCuda();

    Compute::Conv2DBackward(dx, dFilter, dy, x, filter, strideRow, strideCol,
                            rowPadding, colPadding, dilationRow, dilationCol);

    auto dxDataCuda = dx.GetDataCopy();
    auto dFilterDataCuda = dFilter.GetDataCopy();

    for (int i = 0; i < dx.Size(); ++i)
    {
        if (!std::isnan(dxDataHost[i]) && !std::isnan(dxDataCuda[i]))
            CHECK(std::abs(dxDataHost[i] - dxDataCuda[i]) < 0.01f);
        if (print)
            std::cout
                << "host[" << i
                << "] = " << dxDataHost[i] << "  cuda["
                << i << "] = " << dxDataCuda[i] << std::endl;
    }

    std::cout << "Filter" << std::endl;
    for (int i = 0; i < filter.Size(); ++i)
    {
        if (!std::isnan(dFilterDataHost[i]) && !std::isnan(dFilterDataCuda[i]))
            CHECK(std::abs(dFilterDataHost[i] - dFilterDataCuda[i]) < 0.01f);
        if (print)
            std::cout << "host[" << i
                << "] = " << dFilterDataHost[i]
                << "  cuda[" << i << "] = " << dFilterDataCuda[i]
                << std::endl;
    }
}
}
