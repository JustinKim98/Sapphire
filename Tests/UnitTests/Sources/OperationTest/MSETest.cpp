// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/MSETest.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <TestUtil.hpp>
#include <iostream>
#include <random>
#include <doctest.h>

namespace Sapphire::Test
{
void TestMSE(bool print)
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");

    const int inputs = 10;

    const Shape xShape = Shape({ 3, 6, 7, inputs });

    const auto batchSize = xShape.GetNumUnits(1);

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
    const auto gpuForwardPtr = gpuLoss.GetData();
    gpuLoss.SetGradient(backwardData);
    Optimizer::SGD sgd(0.0f);
    ModelManager::CurModel().SetOptimizer(&sgd);
    ModelManager::CurModel().BackProp(gpuLoss);
    const auto gpuBackwardPtr = x.GetGradient();

    x.ToHost();
    label.ToHost();
    const auto hostLoss = NN::Loss::MSE(x, label);
    const auto hostForwardPtr = hostLoss.GetData();
    hostLoss.SetGradient(backwardData);
    ModelManager::CurModel().BackProp(hostLoss);
    const auto hostBackwardPtr = x.GetGradient();

    CHECK(gpuLoss.GetShape().Cols() == 1);
    CHECK(gpuLoss.GetShape().Rows() == 1);

    if (print)
    {
        std::cout << "MSE Forward (Cuda)" << std::endl;
        std::cout << gpuForwardPtr[0] << " " << std::endl;;

        std::cout << "MSE Backward (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int i = 0; i < inputs; ++i)
                std::cout << gpuBackwardPtr[batchIdx * inputs + i] << " ";
            std::cout << std::endl;
        }

        std::cout << "MSE Forward (Host)" << std::endl;
        std::cout << hostForwardPtr[0] << " " << std::endl;

        std::cout << "MSE Backward(Host)" << std::endl;
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
