// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/MeanTest.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/operations/Forward/MathForward.hpp>
#include <iostream>
#include <random>
#include "doctest.h"

namespace Sapphire::Test
{
void TestMean(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dist(1, 10);

    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");

    const Shape xShape = Shape({ dist(gen), dist(gen), dist(gen)
    });

    Tensor x(xShape, gpu, Type::Dense);

    Initialize::Initialize(
        x, std::make_unique<Initialize::Normal>(5.0f, 1.0f));

    const int dim = static_cast<int>(dist(gen) % 3);

    x.ToCuda();
    auto yGpu = NN::Functional::MeanOp(x, dim);
    yGpu.ToHost();
    x.ToHost();
    const auto yHost = NN::Functional::MeanOp(x, dim);

    const auto yForwardGpu = yGpu.GetForwardDataCopy();
    const auto yForwardHost = yHost.GetForwardDataCopy();
    const auto yShape = yGpu.GetShape();

    for (std::size_t i = 0; i < yShape.Size(); ++i)
        CHECK(std::abs(yForwardGpu[i] - yForwardHost[i]) <
        std::numeric_limits<float>::epsilon());

    x.ToCuda();
    yGpu.ToCuda();
    Initialize::InitializeBackwardData(yGpu,
                                       std::make_unique<Initialize::Normal>(
                                           0.0f, 10.0f));
    ModelManager::GetCurrentModel().BackProp(yGpu);
    x.ToHost();

    const auto xBackwardGpu = x.GetBackwardDataCopy();

    Initialize::InitializeBackwardData(
        x, std::make_unique<Initialize::Zeros>());
    ModelManager::GetCurrentModel().BackProp(yHost);
    x.ToHost();

    const auto xBackwardHost = x.GetBackwardDataCopy();

    for (std::size_t i = 0; i < xShape.Size(); ++i)
        CHECK(std::abs(xBackwardGpu[i] - xBackwardHost[i]) <
        std::numeric_limits<float>::epsilon());

    if (print)
    {
        std::cout << "YForward (GPU)" << std::endl;
        for (std::size_t i = 0; i < yShape.Size(); ++i)
            std::cout << yForwardGpu[i] << std::endl;

        std::cout << "YForward (Host)" << std::endl;
        for (std::size_t i = 0; i < yShape.Size(); ++i)
            std::cout << yForwardHost[i] << std::endl;

        std::cout << "XBackward (GPU)" << std::endl;
        for (std::size_t i = 0; i < xShape.Size(); ++i)
            std::cout << xBackwardGpu[i] << std::endl;

        std::cout << "XBackward (Host)" << std::endl;
        for (std::size_t i = 0; i < xShape.Size(); ++i)
            std::cout << xBackwardHost[i] << std::endl;
    }

    ModelManager::GetCurrentModel().Clear();
}
}
