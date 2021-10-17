// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/MathTest.hpp>
#include <Sapphire/operations/Forward/MathForward.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <iostream>

namespace Sapphire::Test
{
void TestMultiply(bool print)
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");
    const auto m = 10;
    const auto n = 10;
    const auto k = 10;

    const Shape shapeA = Shape({ 1, m, k });
    const Shape shapeB = Shape({ 1, k, n });

    Tensor inputA(shapeA, gpu, Type::Dense);
    Tensor inputB(shapeB, gpu, Type::Dense);
    inputA.SetMode(DeviceType::Host);
    inputB.SetMode(DeviceType::Host);

    Initialize::Initialize(
        inputA, std::make_unique<Initialize::Normal>(0.0f, 10.0f));
    Initialize::Initialize(
        inputB, std::make_unique<Initialize::Normal>(0.0f, 10.0f));

    inputA.ToCuda();
    inputB.ToCuda();
    auto y = NN::Functional::MulOp(inputA, inputB);

    y.ToHost();
    const auto forwardDataPtr = y.GetDataCopy();
    const auto outputRows = y.GetShape().Rows();
    const auto outputCols = y.GetShape().Cols();

    Initialize::InitializeBackwardData(
        y, std::make_unique<Initialize::Normal>(0.0f, 10.0f));
    y.ToCuda();
    ModelManager::CurModel().BackProp(y);

    inputA.ToHost();
    inputB.ToHost();

    const auto backwardDataPtrA = inputA.GetBackwardDataCopy();
    const auto backwardDataPtrB = inputB.GetBackwardDataCopy();

    if (print)
    {
        std::cout << "Y Forward" << std::endl;
        for (int i = 0; i < outputRows; ++i)
        {
            for (int j = 0; j < outputCols; ++j)
            {
                std::cout << forwardDataPtr[i * outputCols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputA Backward" << std::endl;
        for (int i = 0; i < inputA.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputA.GetShape().Cols(); ++j)
            {
                std::cout << backwardDataPtrA[i * inputA.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputB Backward" << std::endl;
        for (int i = 0; i < inputB.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputB.GetShape().Cols(); ++j)
            {
                std::cout << backwardDataPtrB[i * inputB.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }
    }

    ModelManager::CurModel().Clear();
}

void TestAdd(bool print)
{
    const CudaDevice gpu(0, "cuda0");
    const auto m = 10;
    const auto n = 10;
    const auto k = 10;

    const Shape shapeA = Shape({ 1, m, k });
    const Shape shapeB = Shape({ 1, k, n });

    Tensor inputA(shapeA, gpu, Type::Dense);
    Tensor inputB(shapeB, gpu, Type::Dense);

    inputA.ToCuda();
    inputB.ToCuda();

    Initialize::Initialize(inputA,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));
    Initialize::Initialize(inputB,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));

    auto y = NN::Functional::AddOp(inputA, inputB);

    y.ToHost();
    const auto forwardDataPtr = y.GetDataCopy();
    const auto outputRows = y.GetShape().Rows();
    const auto outputCols = y.GetShape().Cols();

    Initialize::InitializeBackwardData(
        y, std::make_unique<Initialize::Normal>(0.0f, 1.0f));
    y.ToCuda();
    ModelManager::CurModel().BackProp(y);

    inputA.ToHost();
    inputB.ToHost();

    const auto backwardDataPtrA = inputA.GetBackwardDataCopy();
    const auto backwardDataPtrB = inputB.GetBackwardDataCopy();

    if (print)
    {
        std::cout << "Y Forward" << std::endl;
        for (int i = 0; i < outputRows; ++i)
        {
            for (int j = 0; j < outputCols; ++j)
            {
                std::cout << forwardDataPtr[i * outputCols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputA Backward" << std::endl;
        for (int i = 0; i < inputA.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputA.GetShape().Cols(); ++j)
            {
                std::cout << backwardDataPtrA[i * inputA.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputB Backward" << std::endl;
        for (int i = 0; i < inputB.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputB.GetShape().Cols(); ++j)
            {
                std::cout << backwardDataPtrB[i * inputB.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }
    }
    ModelManager::CurModel().Clear();
}
}
