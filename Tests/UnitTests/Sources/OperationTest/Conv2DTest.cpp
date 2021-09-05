// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/Conv2DTest.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Conv2D.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <iostream>
#include <doctest/doctest.h>
#include <Sapphire/operations/Forward/Linear.hpp>

namespace Sapphire::Test
{
void TestConv2D()
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");
    const int inputChannels = 3;
    const int outputChannels = 3;
    const int inputRows = 4;
    const int inputCols = 4;

    const auto inputSize = std::make_pair(inputRows, inputCols);
    const auto kernelSize = std::make_pair(3, 3);
    const auto stride = std::make_pair(1, 1);
    const auto dilation = std::make_pair(1, 1);
    const auto padSize = std::make_pair(1, 1);

    NN::Conv2D conv2D(inputChannels, outputChannels, inputSize, kernelSize,
                      stride, padSize, dilation, false,
                      Util::SharedPtr<Optimizer::SGD>::Make(0.1f),
                      std::make_unique<Initialize::Ones>(),
                      std::make_unique<Initialize::Ones>(), gpu);

    Tensor input(Shape({ inputChannels, 4, 4 }), gpu, Type::Dense);
    Initialize::Initialize(input, std::make_unique<Initialize::Ones>());
    input.ToCuda();
    auto output = conv2D(input);
    output.ToHost();

    const auto forwardDataPtr = output.GetForwardDataCopy();
    const auto outputRows = output.GetShape().Rows();
    const auto outputCols = output.GetShape().Cols();

    for (std::size_t i = 0; i < outputRows; ++i)
    {
        for (std::size_t j = 0; j < outputCols; ++j)
        {
            std::cout << forwardDataPtr[i * outputCols + j] << " ";
        }
        std::cout << std::endl;
    }

    Initialize::InitializeBackwardData(output,
                                       std::make_unique<Initialize::Ones>());

    output.ToCuda();
    ModelManager::GetCurrentModel().BackProp(output);

    input.ToHost();
    const auto backwardDataPtr = input.GetBackwardDataCopy();
    for (std::size_t i = 0; i < inputRows; ++i)
    {
        for (std::size_t j = 0; j < inputCols; ++j)
        {
            std::cout << backwardDataPtr[i * outputCols + j] << " ";
        }
        std::cout << std::endl;
    }

    ModelManager::GetCurrentModel().Clear();
}
}
