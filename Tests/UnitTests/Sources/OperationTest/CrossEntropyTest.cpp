// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/CrossEntropyTest.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Loss/CrossEntropy.hpp>
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

    const CudaDevice gpu(0, "cuda0");

    const int inputs = 10;

    const Shape xShape = Shape({2,6,3, inputs });

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
    gpuLoss.SetGradient(backwardData);
    ModelManager::CurModel().BackProp(gpuLoss);
    const auto gpuBackwardPtr = x.GetGradient();

    x.SetGradient(std::vector<float>(x.Size(), 0.0f));

    x.ToHost();
    label.ToHost();
    const auto hostLoss = NN::Loss::CrossEntropy(x, label);
    const auto hostForwardPtr = hostLoss.GetData();
    hostLoss.SetGradient(backwardData);
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
}
}
