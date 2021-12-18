// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/MathTest.hpp>
#include <TestUtil.hpp>
#include <Sapphire/operations/Forward/Functional/MathForward.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <iostream>
#include <random>
#include <doctest.h>

namespace Sapphire::Test
{
void TestMatMul(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-5.0f, 5.0f);

    const CudaDevice gpu(0, "cuda0");
    constexpr auto m = 3;
    constexpr auto n = 4;
    constexpr auto k = 2;

    const Shape shapeA = Shape({ 1, m, k });
    const Shape shapeB = Shape({ 1, k, n });

    ModelManager::AddModel("TestMatMul");
    ModelManager::SetCurrentModel("TestMatMul");

    Tensor inputA(shapeA, gpu, Type::Dense);
    Tensor inputB(shapeB, gpu, Type::Dense);

    inputA.ToCuda();
    inputB.ToCuda();

    Initialize::Initialize(inputA,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));
    Initialize::Initialize(inputB,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));

    Optimizer::SGD sgd(0.1f);
    ModelManager::CurModel().SetOptimizer(&sgd);

    auto y = F::MatMul(inputA, inputB);
    std::vector<float> backwardData(y.GetShape().Size());
    for (auto& elem : backwardData)
        elem = dist(gen);
    y.LoadGradient(backwardData);
    const auto forwardDataCuda = y.GetData();
    ModelManager::CurModel().BackProp(y);

    const auto gradientACuda = inputA.GetGradient();
    const auto gradientBCuda = inputB.GetGradient();

    inputA.ToHost();
    inputB.ToHost();

    Initialize::InitializeGradient(inputA,
                                   std::make_unique<Initialize::Zeros>());
    Initialize::InitializeGradient(inputB,
                                   std::make_unique<Initialize::Zeros>());

    y = F::MatMul(inputA, inputB);
    y.LoadGradient(backwardData);
    const auto forwardDataHost = y.GetData();
    ModelManager::CurModel().BackProp(y);

    const auto gradientAHost = inputA.GetGradient();
    const auto gradientBHost = inputB.GetGradient();

    const auto outputRows = y.GetShape().Rows();
    const auto outputCols = y.GetShape().Cols();
    
    for (int i = 0; i < y.GetShape().Size(); ++i)
        CHECK(TestEquality(forwardDataCuda[i], forwardDataHost[i]));
    for (int i = 0; i < inputA.GetShape().Size(); ++i)
        CHECK(TestEquality(gradientACuda[i], gradientAHost[i]));
    for (int i = 0; i < inputB.GetShape().Size(); ++i)
        CHECK(TestEquality(gradientBCuda[i], gradientBHost[i]));

    if (print)
    {
        std::cout << "Y Forward" << std::endl;
        for (int i = 0; i < outputRows; ++i)
        {
            for (int j = 0; j < outputCols; ++j)
            {
                std::cout << forwardDataCuda[i * outputCols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputA Backward" << std::endl;
        for (int i = 0; i < inputA.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputA.GetShape().Cols(); ++j)
            {
                std::cout << gradientACuda[i * inputA.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputB Backward" << std::endl;
        for (int i = 0; i < inputB.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputB.GetShape().Cols(); ++j)
            {
                std::cout << gradientBCuda[i * inputB.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }
    }
    ModelManager::CurModel().Clear();
    Util::ResourceManager::ClearAll();
}

void TestAdd(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-1.0f, 1.0f);

    const CudaDevice gpu(0, "cuda0");
    constexpr auto N = 100;

    const Shape shapeA = Shape({ N });
    const Shape shapeB = Shape({ N });

    ModelManager::AddModel("TestAdd");
    ModelManager::SetCurrentModel("TestAdd");

    Tensor inputA(shapeA, gpu, Type::Dense);
    Tensor inputB(shapeB, gpu, Type::Dense);

    inputA.ToCuda();
    inputB.ToCuda();

    Initialize::Initialize(inputA,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));
    Initialize::Initialize(inputB,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));

    Optimizer::SGD sgd(0.1f);
    ModelManager::CurModel().SetOptimizer(&sgd);

    auto y = F::Add(inputA, inputB);
    std::vector<float> backwardData(y.GetShape().Size());
    for (auto& elem : backwardData)
        elem = dist(gen);
    y.LoadGradient(backwardData);
    const auto forwardDataCuda = y.GetData();
    ModelManager::CurModel().BackProp(y);

    const auto gradientACuda = inputA.GetGradient();
    const auto gradientBCuda = inputB.GetGradient();

    inputA.ToHost();
    inputB.ToHost();

    Initialize::InitializeGradient(inputA,
                                   std::make_unique<Initialize::Zeros>());
    Initialize::InitializeGradient(inputB,
                                   std::make_unique<Initialize::Zeros>());

    y = F::Add(inputA, inputB);
    y.LoadGradient(backwardData);
    const auto forwardDataHost = y.GetData();
    ModelManager::CurModel().BackProp(y);

    const auto gradientAHost = inputA.GetGradient();
    const auto gradientBHost = inputB.GetGradient();

    const auto outputRows = y.GetShape().Rows();
    const auto outputCols = y.GetShape().Cols();

    for (int i = 0; i < y.GetShape().Size(); ++i)
        CHECK(TestEquality(forwardDataCuda[i], forwardDataHost[i]));
    for (int i = 0; i < inputA.GetShape().Size(); ++i)
        CHECK(TestEquality(gradientACuda[i], gradientAHost[i]));
    for (int i = 0; i < inputB.GetShape().Size(); ++i)
        CHECK(TestEquality(gradientBCuda[i], gradientBHost[i]));

    if (print)
    {
        std::cout << "Y Forward" << std::endl;
        for (int i = 0; i < outputRows; ++i)
        {
            for (int j = 0; j < outputCols; ++j)
            {
                std::cout << forwardDataCuda[i * outputCols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputA Backward" << std::endl;
        for (int i = 0; i < inputA.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputA.GetShape().Cols(); ++j)
            {
                std::cout << gradientACuda[i * inputA.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputB Backward" << std::endl;
        for (int i = 0; i < inputB.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputB.GetShape().Cols(); ++j)
            {
                std::cout << gradientBCuda[i * inputB.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }
    }
    ModelManager::CurModel().Clear();
    Util::ResourceManager::ClearAll();
}

void TestSub(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-1.0f, 1.0f);

    const CudaDevice gpu(0, "cuda0");
    constexpr auto N = 100;

    const Shape shapeA = Shape({ N });
    const Shape shapeB = Shape({ N });

    ModelManager::AddModel("TestSub");
    ModelManager::SetCurrentModel("TestSub");

    Tensor inputA(shapeA, gpu, Type::Dense);
    Tensor inputB(shapeB, gpu, Type::Dense);

    inputA.ToCuda();
    inputB.ToCuda();

    Initialize::Initialize(inputA,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));
    Initialize::Initialize(inputB,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));

    Optimizer::SGD sgd(0.1f);
    ModelManager::CurModel().SetOptimizer(&sgd);

    auto y = F::Sub(inputA, inputB);
    std::vector<float> backwardData(y.GetShape().Size());
    for (auto& elem : backwardData)
        elem = dist(gen);
    y.LoadGradient(backwardData);
    const auto forwardDataCuda = y.GetData();
    ModelManager::CurModel().BackProp(y);

    const auto gradientACuda = inputA.GetGradient();
    const auto gradientBCuda = inputB.GetGradient();

    inputA.ToHost();
    inputB.ToHost();

    Initialize::InitializeGradient(inputA,
                                   std::make_unique<Initialize::Zeros>());
    Initialize::InitializeGradient(inputB,
                                   std::make_unique<Initialize::Zeros>());

    y = F::Sub(inputA, inputB);
    y.LoadGradient(backwardData);
    const auto forwardDataHost = y.GetData();
    ModelManager::CurModel().BackProp(y);

    const auto gradientAHost = inputA.GetGradient();
    const auto gradientBHost = inputB.GetGradient();

    const auto outputRows = y.GetShape().Rows();
    const auto outputCols = y.GetShape().Cols();

    for (int i = 0; i < y.GetShape().Size(); ++i)
        CHECK(TestEquality(forwardDataCuda[i], forwardDataHost[i]));
    for (int i = 0; i < inputA.GetShape().Size(); ++i)
        CHECK(TestEquality(gradientACuda[i], gradientAHost[i]));
    for (int i = 0; i < inputB.GetShape().Size(); ++i)
        CHECK(TestEquality(gradientBCuda[i], gradientBHost[i]));

    if (print)
    {
        std::cout << "Y Forward" << std::endl;
        for (int i = 0; i < outputRows; ++i)
        {
            for (int j = 0; j < outputCols; ++j)
            {
                std::cout << forwardDataCuda[i * outputCols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputA Backward" << std::endl;
        for (int i = 0; i < inputA.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputA.GetShape().Cols(); ++j)
            {
                std::cout << gradientACuda[i * inputA.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputB Backward" << std::endl;
        for (int i = 0; i < inputB.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputB.GetShape().Cols(); ++j)
            {
                std::cout << gradientBCuda[i * inputB.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }
    }
    ModelManager::CurModel().Clear();
    Util::ResourceManager::ClearAll();
}


void TestDot(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-1.0f, 1.0f);

    const CudaDevice gpu(0, "cuda0");
    constexpr auto N = 100;

    const Shape shapeA = Shape({ N });
    const Shape shapeB = Shape({ N });

    Tensor inputA(shapeA, gpu, Type::Dense);
    Tensor inputB(shapeB, gpu, Type::Dense);

    ModelManager::AddModel("TestDot");
    ModelManager::SetCurrentModel("TestDot");

    inputA.ToCuda();
    inputB.ToCuda();

    Initialize::Initialize(inputA,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));
    Initialize::Initialize(inputB,
                           std::make_unique<Initialize::Normal>(0.0f, 10.0f));

    Optimizer::SGD sgd(0.1f);
    ModelManager::CurModel().SetOptimizer(&sgd);

    auto y = F::Dot(inputA, inputB);
    std::vector<float> backwardData(y.GetShape().Size());
    for (auto& elem : backwardData)
        elem = dist(gen);
    y.LoadGradient(backwardData);
    const auto forwardDataCuda = y.GetData();
    ModelManager::CurModel().BackProp(y);

    const auto gradientACuda = inputA.GetGradient();
    const auto gradientBCuda = inputB.GetGradient();

    inputA.ToHost();
    inputB.ToHost();

    Initialize::InitializeGradient(inputA,
                                   std::make_unique<Initialize::Zeros>());
    Initialize::InitializeGradient(inputB,
                                   std::make_unique<Initialize::Zeros>());

    y = F::Dot(inputA, inputB);
    y.LoadGradient(backwardData);
    const auto forwardDataHost = y.GetData();
    ModelManager::CurModel().BackProp(y);

    const auto gradientAHost = inputA.GetGradient();
    const auto gradientBHost = inputB.GetGradient();

    const auto outputRows = y.GetShape().Rows();
    const auto outputCols = y.GetShape().Cols();

    for (int i = 0; i < y.GetShape().Size(); ++i)
        CHECK(TestEquality(forwardDataCuda[i], forwardDataHost[i]));
    for (int i = 0; i < inputA.GetShape().Size(); ++i)
        CHECK(TestEquality(gradientACuda[i], gradientAHost[i]));
    for (int i = 0; i < inputB.GetShape().Size(); ++i)
        CHECK(TestEquality(gradientBCuda[i], gradientBHost[i]));

    if (print)
    {
        std::cout << "Y Forward" << std::endl;
        for (int i = 0; i < outputRows; ++i)
        {
            for (int j = 0; j < outputCols; ++j)
            {
                std::cout << forwardDataCuda[i * outputCols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputA Backward" << std::endl;
        for (int i = 0; i < inputA.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputA.GetShape().Cols(); ++j)
            {
                std::cout << gradientACuda[i * inputA.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "inputB Backward" << std::endl;
        for (int i = 0; i < inputB.GetShape().Rows(); ++i)
        {
            for (int j = 0; j < inputB.GetShape().Cols(); ++j)
            {
                std::cout << gradientBCuda[i * inputB.GetShape().Cols() + j]
                    << " ";
            }
            std::cout << std::endl;
        }
    }
    ModelManager::CurModel().Clear();
    Util::ResourceManager::ClearAll();
}
}
