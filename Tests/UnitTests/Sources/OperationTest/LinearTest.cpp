// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/LinearTest.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <TestUtil.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <doctest/doctest.h>
#include <iostream>
#include <random>

namespace Sapphire::Test
{
void TestLinear(bool print)
{
    //! Initialize model hyperparameters
    constexpr int batchSize = 1;
    constexpr int inputs = 5;
    constexpr int outputs = 3;

    //! Initialize random input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-10.0f, 10.0f);
    std::vector<float> forwardData(batchSize * inputs);
    std::vector<float> backwardData(batchSize * outputs);

    for (auto& data : forwardData)
        data = dist(gen);
    for (auto& data : backwardData)
        data = dist(gen);

    //! Initialize model and device
    ModelManager::AddModel("linear test model");
    ModelManager::SetCurrentModel("linear test model");
    const CudaDevice gpu(0, "cuda0");

    Tensor input(Shape({ batchSize, 1, inputs }), gpu, Type::Dense, true);
    input.LoadData(forwardData);

    //! Declare layer to test
    NN::Linear linear(inputs, outputs);
    auto weightData = linear.GetWeight().GetData();
    auto biasData = linear.GetBias().GetData();

    //! Setup optimizer
    Optimizer::SGD sgd(0.01f);
    ModelManager::CurModel().SetOptimizer(&sgd);

    //! Test the operation using cuda
    linear.ToCuda();
    input.ToCuda();
    auto gpuOutput = linear(input);
    const auto gpuForwardData = gpuOutput.GetData();
    gpuOutput.SetGradient(backwardData);
    ModelManager::CurModel().BackProp(gpuOutput);
    const auto gpuGradientData = input.GetGradient();
    const auto gpuWeightData = linear.GetWeight().GetData();
    const auto gpuBiasData = linear.GetBias().GetData();

    //! Reset backward Gradient
    ModelManager::CurModel().Clear();
    linear.GetWeight().LoadData(weightData);
    linear.GetBias().LoadData(biasData);

    //! Test the operation on the host
    linear.ToHost();
    input.ToHost();
    const auto hostOutput = linear(input);
    const auto hostForwardData = hostOutput.GetData();
    hostOutput.SetGradient(backwardData);
    ModelManager::CurModel().BackProp(hostOutput);
    const auto hostGradientData = input.GetGradient();
    const auto hostWeightData = linear.GetWeight().GetData();
    const auto hostBiasData = linear.GetBias().GetData();

    if (print)
    {
        std::cout << "Linear forward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < outputs; ++i)
            {
                std::cout << hostForwardData[batchIdx * outputs + i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Linear forward result (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < outputs; ++i)
            {
                std::cout << gpuForwardData[batchIdx * outputs + i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Linear backward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < outputs; ++i)
            {
                std::cout << hostGradientData[batchIdx * outputs + i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Linear backward result (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < outputs; ++i)
            {
                std::cout << gpuGradientData[batchIdx * outputs + i] << " ";
            }
            std::cout << std::endl;
        }
    }

    //! Check Equalities for all data
    for (int i = 0; i < batchSize * outputs; ++i)
        CHECK(TestEquality(hostForwardData[i], gpuForwardData[i]));
    for (int i = 0; i < batchSize * inputs; ++i)
        CHECK(TestEquality(hostGradientData[i], gpuGradientData[i]));
    for (std::size_t i = 0; i < weightData.size(); ++i)
        CHECK(TestEquality(hostWeightData[i], gpuWeightData[i]));
    for (std::size_t i = 0; i < biasData.size(); ++i)
        CHECK(TestEquality(hostBiasData[i], gpuBiasData[i]));

    ModelManager::CurModel().Clear();
}

//! Test simple weight decay
void TestLinearTraining(bool printData)
{
    constexpr int epochs = 100;
    constexpr int inputFeatureSize = 10;
    constexpr int outputFeatureSize = 5;
    constexpr float learningRate = 0.0001f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-10.0f, 10.0f);

    ModelManager::AddModel("SimpleLinearModel");
    ModelManager::SetCurrentModel("SimpleLinearModel");

    const CudaDevice gpu(0, "cuda0");

    NN::Linear linear(inputFeatureSize, outputFeatureSize);

    Tensor x(Shape({ inputFeatureSize }), gpu, Type::Dense, true);
    Tensor label(Shape({ outputFeatureSize }), gpu, Type::Dense, true);

    Optimizer::SGD sgd(learningRate);
    ModelManager::CurModel().SetOptimizer(&sgd);

    std::vector<float> labelData(outputFeatureSize);
    std::vector<float> xData(inputFeatureSize);

    for (auto& data : labelData)
        data = dist(gen);
    for (auto& data : xData)
        data = dist(gen);

    for (int i = 0; i < epochs; ++i)
    {
        x.LoadData(xData);
        label.LoadData(labelData);
        auto tensor = linear(x);
        const auto loss = NN::Loss::MSE(tensor, label);
        if (i % 10 == 0)
        {
            if (printData)
            {
                const auto yDataCopy = tensor.GetData();
                const auto labelDataCopy = label.GetData();
                for (const auto& elem : yDataCopy)
                    std::cout << elem << " ";
                std::cout << std::endl;
            }
            const auto lossData = loss.GetData();
            std::cout << "epoch: " << i << " loss : " << lossData[0]
                << std::endl;
        }
        ModelManager::CurModel().BackProp(loss);
        ModelManager::CurModel().Clear();
        if (i % 10 == 0)
            Util::ResourceManager::Clean();
    }
    Util::ResourceManager::ClearAll();
}
} // namespace Sapphire::Test
