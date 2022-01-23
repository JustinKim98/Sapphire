// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/CrossEntropyTest.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/operations/Forward/Functional/Softmax.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Loss/CrossEntropy.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/operations/Forward/Functional/ReLU.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <TestUtil.hpp>
#include <iostream>
#include <random>
#include <doctest.h>

namespace Sapphire::Test
{
void TestCrossEntropy(bool print)
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const DeviceInfo gpu(0, "cuda0");

    const int inputs = 10;

    const Shape xShape = Shape({ 2, 6, 3, inputs });

    Tensor x(xShape, gpu, Type::Dense);
    Tensor label(xShape, gpu, Type::Dense);
    x.ToHost();
    label.ToHost();

    const auto batchSize = xShape.GetNumUnits(1);

    const std::vector backwardData(batchSize, 0.0f);

    Initialize::Initialize(x, std::make_unique<Initialize::Normal>(0.0f, 1.0f));
    Initialize::Initialize(label,
                           std::make_unique<Initialize::Normal>(0.0f, 1.0f));

    x.ToCuda();
    label.ToCuda();
    const auto gpuLoss = NN::Loss::CrossEntropy(x, label);
    const auto lossShape = gpuLoss.GetShape();
    const auto gpuForwardPtr = gpuLoss.GetData();
    gpuLoss.LoadGradient(backwardData);
    ModelManager::CurModel().BackProp(gpuLoss);
    const auto gpuBackwardPtr = x.GetGradient();

    x.LoadGradient(std::vector<float>(x.Size(), 0.0f));

    x.ToHost();
    label.ToHost();
    const auto hostLoss = NN::Loss::CrossEntropy(x, label);
    const auto hostForwardPtr = hostLoss.GetData();
    hostLoss.LoadGradient(backwardData);

    Optimizer::SGD sgd(0.0f);
    ModelManager::CurModel().SetOptimizer(&sgd);
    ModelManager::CurModel().BackProp(hostLoss);
    const auto hostBackwardPtr = x.GetGradient();

    CHECK(gpuLoss.GetShape().At(-1) == 1);
    CHECK(gpuLoss.GetShape().At(-2) == batchSize);

    if (print)
    {
        std::cout << " CrossEntropy Forward (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << gpuForwardPtr[batchIdx] << " ";
        }
        std::cout << std::endl;

        std::cout << "CrossEntropy Backward (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < inputs; ++i)
                std::cout << gpuBackwardPtr[batchIdx * inputs + i] << " ";
            std::cout << std::endl;
        }

        std::cout << "CrossEntropy Forward (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << hostForwardPtr[batchIdx] << " ";
        }
        std::cout << std::endl;

        std::cout << "CrossEntropy Backward(Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < inputs; ++i)
                std::cout << hostBackwardPtr[batchIdx * inputs + i] << " ";
            std::cout << std::endl;
        }
    }

    for (int i = 0; i < gpuLoss.GetShape().Size(); ++i)
        CHECK(TestEquality(hostForwardPtr[i], gpuForwardPtr[i]));

    for (int i = 0; i < x.GetShape().Size(); ++i)
        CHECK(TestEquality(hostBackwardPtr[i], gpuBackwardPtr[i]));

    ModelManager::CurModel().Clear();
    Util::ResourceManager::ClearAll();
}

//! Test simple weight decay
void TestCrossEntropyTraining(bool printData)
{
    constexpr int epochs = 100;
    constexpr int inputFeatureSize = 10;
    constexpr int outputFeatureSize = 5;
    constexpr float learningRate = 0.0001f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-1.0f, 1.0f);

    ModelManager::AddModel("SimpleLinearModel");
    ModelManager::SetCurrentModel("SimpleLinearModel");

    const DeviceInfo gpu(0, "cuda0");

    NN::Linear linear(inputFeatureSize, outputFeatureSize);
    NN::Linear linear1(outputFeatureSize, outputFeatureSize);

    Tensor x(Shape({ inputFeatureSize }), gpu, Type::Dense, true);
    Tensor label(Shape({ outputFeatureSize }), gpu, Type::Dense, true);

    // x.ToHost();
    // label.ToHost();
    // linear.ToHost();

    Optimizer::SGD sgd(learningRate);
    ModelManager::CurModel().SetOptimizer(&sgd);

    std::vector<float> labelData(outputFeatureSize);
    std::vector<float> xData(inputFeatureSize);

    for (auto& data : labelData)
        data = 0.0f;
    for (auto& data : xData)
        data = dist(gen);

    labelData[3] = 1.0f;

    for (int i = 0; i < epochs; ++i)
    {
        x.LoadData(xData);
        label.LoadData(labelData);
        auto tensor = F::ReLU(linear(x));
        tensor = linear1(tensor);
        tensor = F::SoftMax(tensor);
        const auto loss = NN::Loss::CrossEntropy(tensor, label);
        // const auto loss = NN::Loss::MSE(tensor, label);
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
}
