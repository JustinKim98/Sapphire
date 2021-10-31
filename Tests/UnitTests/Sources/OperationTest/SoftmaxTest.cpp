// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/SoftmaxTest.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/Forward/Softmax.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <TestUtil.hpp>
#include <iostream>
#include <doctest/doctest.h>
#include <random>

namespace Sapphire::Test
{
void TestSoftmax(bool print)
{
    ModelManager::AddModel("softmaxTestModel");
    ModelManager::SetCurrentModel("softmaxTestModel");

    const CudaDevice gpu(0, "cuda0");
    constexpr int batchSize = 2;
    constexpr int unitSize = 10;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    std::vector<float> backwardData(batchSize * unitSize);
    for (auto& data : backwardData)
        data = dist(gen);

    NN::Linear linear(10, 10, new Optimizer::SGD(0.0f));

    Tensor input({ batchSize, unitSize }, gpu);
    Tensor weight({ unitSize, unitSize }, gpu, true);
    Tensor bias({ unitSize }, gpu, true);
    input.SetMode(ComputeMode::Host);
    bias.SetMode(ComputeMode::Host);
    weight.SetMode(ComputeMode::Host);

    Initialize::Initialize(input,
                           std::make_unique<Initialize::Normal>(0.0f, 1.0f));
    Initialize::Initialize(weight,
                           std::make_unique<Initialize::Normal>(0.0f, 1.0f));
    Initialize::Initialize(bias,
                           std::make_unique<Initialize::Normal>(0.0f, 1.0f));

    //auto tensor = linear(input, weight, bias);
    auto tensor = NN::SoftMax(input);

    const auto forwardDataHost = tensor.GetData();
    tensor.SetGradient(backwardData);
    ModelManager::CurModel().BackProp(tensor);
    const auto backwardDataHost = input.GetGradient();

    input.ToCuda();
    bias.ToCuda();
    weight.ToCuda();

    Initialize::InitializeBackwardData(input,
                                       std::make_unique<Initialize::Zeros>());

    //tensor = linear(input, weight, bias);
    tensor = NN::SoftMax(input);
    const auto forwardDataCuda = tensor.GetData();
    tensor.SetGradient(backwardData);
    ModelManager::CurModel().BackProp(tensor);
    const auto backwardDataCuda = input.GetGradient();

    if (print)
    {
        std::cout << "Linear + Softmax Forward result (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int unitIdx = 0; unitIdx < unitSize; ++unitIdx)
            {
                std::cout << forwardDataCuda[batchIdx * unitSize + unitIdx]
                    << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Linear + Softmax Forward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int unitIdx = 0; unitIdx < unitSize; ++unitIdx)
            {
                std::cout << forwardDataHost[batchIdx * unitSize + unitIdx]
                    << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Linear + Softmax Backward result (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int unitIdx = 0; unitIdx < unitSize; ++unitIdx)
            {
                std::cout << backwardDataCuda[batchIdx * unitSize + unitIdx]
                    << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Linear + Softmax Backward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int unitIdx = 0; unitIdx < unitSize; ++unitIdx)
            {
                std::cout << backwardDataHost[batchIdx * unitSize + unitIdx]
                    << " ";
            }
            std::cout << std::endl;
        }
    }

    for (int idx = 0; idx < batchSize * unitSize; ++idx)
        CHECK(TestEquality(forwardDataHost[idx], forwardDataCuda[idx]));

    for (int idx = 0; idx < batchSize * unitSize; ++idx)
        CHECK(TestEquality(backwardDataHost[idx], backwardDataCuda[idx]));

    Util::ResourceManager::ClearAll();
}
}
