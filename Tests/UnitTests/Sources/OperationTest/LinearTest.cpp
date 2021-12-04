// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <OperationTest/LinearTest.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <TestUtil.hpp>
#include <iostream>
#include <random>
#include <doctest/doctest.h>

namespace Sapphire::Test
{
void TestLinear(bool print)
{
    constexpr int batchSize = 2;
    constexpr int inputs = 50;
    constexpr int outputs = 30;

    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    std::vector<float> backwardData(batchSize * outputs);
    for (auto& data : backwardData)
        data = dist(gen);

    Tensor input(Shape({ batchSize, 1, inputs }), gpu, Type::Dense);

    input.SetMode(ComputeMode::Host);

    Initialize::Initialize(input,
                           std::make_unique<Initialize::Normal>(0.0f, 1.0f));

    input.ToCuda();

    NN::Linear linear(inputs, outputs);

    auto gpuOutput = linear(input);
    const auto gpuForwardPtr = gpuOutput.GetData();
    gpuOutput.SetGradient(backwardData);

    Optimizer::SGD sgd(0.0f);
    ModelManager::CurModel().SetOptimizer(&sgd);
    ModelManager::CurModel().BackProp(gpuOutput);
    const auto gpuBackwardPtr = input.GetGradient();

    input.ToHost();

    Initialize::InitializeBackwardData(input,
                                       std::make_unique<Initialize::Zeros>());

    NN::Linear linearHost(inputs, outputs);
    const auto hostOutput = linearHost(input);
    const auto hostForwardPtr = hostOutput.GetData();
    hostOutput.SetGradient(backwardData);
    ModelManager::CurModel().BackProp(hostOutput);
    const auto hostBackwardPtr = input.GetGradient();

    if (print)
    {
        std::cout << "Linear forward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < outputs; ++i)
            {
                std::cout << hostForwardPtr[batchIdx * outputs + i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Linear backward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < outputs; ++i)
            {
                std::cout << hostBackwardPtr[batchIdx * outputs + i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Linear forward result (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < outputs; ++i)
            {
                std::cout << gpuForwardPtr[batchIdx * outputs + i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Linear backward result (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < outputs; ++i)
            {
                std::cout << gpuBackwardPtr[batchIdx * outputs + i] << " ";
            }
            std::cout << std::endl;
        }
    }

    for (int i = 0; i < gpuOutput.GetShape().Size(); ++i)
        CHECK(TestEquality(hostForwardPtr[i], gpuForwardPtr[i]));

    for (int i = 0; i < input.GetShape().Size(); ++i)
        CHECK(TestEquality(hostBackwardPtr[i], gpuBackwardPtr[i]));

    ModelManager::CurModel().Clear();
}
} // namespace Sapphire::Test
