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
    const int batchSize = 2;
    const int inputs = 100;
    const int outputs = 100;

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
    Tensor weight(Shape({ inputs, outputs }), gpu, Type::Dense);
    Tensor bias(Shape({ 1, outputs }), gpu, Type::Dense);

    Initialize::Initialize(input,
                           std::make_unique<Initialize::Normal>(0.0f, 1.0f));
    Initialize::Initialize(weight,
                           std::make_unique<Initialize::Normal>(0.0f, 1.0f));

    input.ToCuda();
    weight.ToCuda();
    bias.ToCuda();

    NN::Linear linear(inputs, outputs,
                      Util::SharedPtr<Optimizer::SGD>::Make(0.0f),
                      gpu);

    auto gpuOutput = linear(input, weight, bias);
    const auto gpuForwardPtr = gpuOutput.GetForwardDataCopy();
    gpuOutput.SetBackwardData(backwardData);
    ModelManager::GetCurrentModel().BackProp(gpuOutput);
    const auto gpuBackwardPtr = input.GetBackwardDataCopy();

    input.ToHost();
    weight.ToHost();
    bias.ToHost();

    Initialize::InitializeBackwardData(input,
                                       std::make_unique<Initialize::Zeros>());

    NN::Linear linearHost(inputs, outputs,
                          Util::SharedPtr<Optimizer::SGD>::Make(0.0f), gpu);
    const auto hostOutput = linearHost(input, weight, bias);
    const auto hostForwardPtr = hostOutput.GetForwardDataCopy();
    hostOutput.SetBackwardData(backwardData);
    ModelManager::GetCurrentModel().BackProp(hostOutput);
    const auto hostBackwardPtr = input.GetBackwardDataCopy();

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

    ModelManager::GetCurrentModel().Clear();
}
} // namespace Sapphire::Test
