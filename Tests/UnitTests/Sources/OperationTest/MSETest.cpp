// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/MSETest.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <TestUtil.hpp>
#include <iostream>
#include <doctest.h>
#include <random>

namespace Sapphire::Test
{
void TestMSE(bool print)
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");

    const auto batchSize = 1;
    const int inputs = 10;

    const Shape xShape = Shape({ batchSize, inputs });

    Tensor x(xShape, gpu, Type::Dense);
    Tensor label(xShape, gpu, Type::Dense);
    x.ToHost();
    label.ToHost();

    const std::vector backwardData = { 0.0f };

    Initialize::Initialize(x, std::make_unique<Initialize::Normal>(0.0f, 1.0f));
    Initialize::Initialize(
        label, std::make_unique<Initialize::Normal>(0.0f, 1.0f));
    x.ToCuda();
    label.ToCuda();

    const auto gpuLoss = NN::Loss::MSE(x, label);
    const auto lossShape = gpuLoss.GetShape();
    const auto gpuForwardPtr = gpuLoss.GetDataCopy();
    gpuLoss.SetBackwardData(backwardData);
    ModelManager::CurModel().BackProp(gpuLoss);
    const auto gpuBackwardPtr = x.GetBackwardDataCopy();

    x.ToHost();
    label.ToHost();
    const auto hostLoss = NN::Loss::MSE(x, label);
    const auto hostForwardPtr = hostLoss.GetDataCopy();
    hostLoss.SetBackwardData(backwardData);
    ModelManager::CurModel().BackProp(hostLoss);
    const auto hostBackwardPtr = x.GetBackwardDataCopy();

    CHECK(gpuLoss.GetShape().Cols() == 1);
    CHECK(gpuLoss.GetShape().Rows() == batchSize);

    if (print)
    {
        std::cout << "MSE Forward (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < 1; ++i)
                std::cout << gpuForwardPtr[batchSize + i] << " ";
            std::cout << std::endl;
        }

        std::cout << "MSE Backward (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < inputs; ++i)
                std::cout << gpuBackwardPtr[batchSize * inputs + i] << " ";
            std::cout << std::endl;
        }

        std::cout << "MSE Forward (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < 1; ++i)
                std::cout << hostForwardPtr[batchSize + i] << " ";
            std::cout << std::endl;
        }

        std::cout << "MSE Backward(Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < inputs; ++i)
                std::cout << hostBackwardPtr[batchSize * inputs + i] << " ";
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
