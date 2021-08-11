// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/Conv2DTest.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>
#include <iostream>

namespace Sapphire::Test
{
void Conv2DForwardTest(bool printForward, bool printBackward)
{
    auto shape = CreateRandomShape(5);
    CudaDevice cuda(0, "cuda0");

    int N = 1;
    int inputHeight = 100;
    int inputWidth = 100;
    int inputChannels = 1;

    int numFilters = 10;
    int filterWidth = 3;
    int filterHeight = 3;
    int strideRow = 1;
    int strideCol = 1;
    int dilationRow = 1;
    int dilationCol = 1;
    int rowPadding = 0;
    int colPadding = 0;

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
    if (printForward)
        for (unsigned i = 0; i < y.DenseTotalLengthHost; ++i)
            std::cout << "forwardData [" << i << "] : " << forwardOutput[i] <<
                std::endl;

    y.SetMode(DeviceType::Cuda);

    Compute::Conv2DBackward(dx, dFilter, dy, x, filter, strideRow, strideCol,
                            dilationRow, dilationCol, rowPadding, colPadding);

    dx.ToHost();

    const auto* backwardOutput = dx.GetDenseHost();
    for (unsigned i = 0; i < dy.DenseTotalLengthHost; ++i)
        std::cout << "backwardData[" << i << "]: " << backwardOutput[i]
            << std::endl;
}
}
