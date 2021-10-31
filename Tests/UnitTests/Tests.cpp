// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <FunctionTest/BroadcastTest.hpp>
#include <FunctionTest/GemmTest.hpp>
#include <Sapphire/Tests/CudaFunctionalityTest.cuh>
#include <BasicsTest/SimpleTest.hpp>
#include <OperationTest/MathTest.hpp>
#include <OperationTest/MeanTest.hpp>
#include <OperationTest/MSETest.hpp>
#include <OperationTest/LinearTest.hpp>
#include <OperationTest/Conv2DTest.hpp>
#include <OperationTest/SoftmaxTest.hpp>
#include <ModelTest/Conv2DModel.hpp>
#include <ModelTest/SimpleLinearModel.hpp>
#include <BasicsTest/TransposeTest.hpp>
#include <TensorTest/TensorFunctionalityTest.hpp>
#include <FunctionTest/Conv2DTest.hpp>
#include <Sapphire/compute/TrigonometricOps.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/ActivationOps.hpp>
#include <GraphTest/GraphFunctionalityTest.hpp>
#include <BasicsTest/ReshapeTest.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <TestUtil.hpp>
#include <iostream>
#include "doctest.h"

#define GraphTest
#define TensorFunctionalityTest
#define BasicsTest
#define ActivationTest
#define GemmTest
#define GemmBroadcastTest
#define InitializeTest
#define ConvolutionTest
#define BasicGraphTest
#define ModelTest

namespace Sapphire::Test
{
TEST_CASE("Simple test")
{
    CHECK(Add(2, 3) == 5);
}

TEST_CASE("Check cuda")
{
#ifdef WITH_CUDA
    PrintCudaVersion();

    SUBCASE("Basic functionality test")
    {
        CHECK(EXIT_SUCCESS == MallocTest());
        std::cout << "Malloc test successful" << std::endl;
    }

    SUBCASE("CublasTest")
    {
        CHECK(EXIT_SUCCESS == CublasTest());
        std::cout << "Cublas test successful" << std::endl;
    }
#endif
}

#ifdef TensorFunctionalityTest
TEST_CASE("TensorFunctionalityTest")
{
    SUBCASE("TensorDataCopy")
    {
        for (int i = 0; i < 5; ++i)
            SendDataBetweenHostDevice(false);
    }

    SUBCASE("TensorDataCopyOnCuda")
    {
        for (int i = 0; i < 5; ++i)
            TensorDataCopyOnCuda(false);
    }

    SUBCASE("TensorDataCopyOnHost")
    {
        for (int i = 0; i < 5; ++i)
            TensorDataCopyOnHost(false);
    }
}
#endif

#ifdef BasicsTest
TEST_CASE("Basics")
{
    const int testLoops = 3;
    SUBCASE("Transpose")
    {
        std::cout << "Transpose" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TransposeTest(false);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Reshape")
    {
        std::cout << "Reshape" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            ReshapeTest(false);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Add")
    {
        std::cout << "Add Test" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithTwoArgumentsWithSameShape(false, 1.0f, Compute::Add);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Sub")
    {
        std::cout << "Sub Test" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithTwoArgumentsWithSameShape(false, 1.0f, Compute::Sub);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Dot")
    {
        std::cout << "Dot Test" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithTwoArgumentsWithSameShape(false, 1.0f, Compute::Dot);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("log")
    {
        std::cout << "Log Test" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithOneArgumentStatic(false, 1.0f, Compute::log);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Inverse")
    {
        std::cout << "Inverse Test" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithOneArgumentStatic(false, 1.0f, Compute::Inverse);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Sin")
    {
        std::cout << "SinTest" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithOneArgumentStatic(false, 1.0f, Compute::Sin);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Cos")
    {
        std::cout << "Cos Test" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithOneArgumentStatic(false, 1.0f, Compute::Cos);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Tan")
    {
        std::cout << "Tan Test" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithOneArgumentStatic(false, 1.0f, Compute::Tan);
        Util::ResourceManager::ClearAll();
    }
}
#endif

#ifdef ActivationTest
TEST_CASE("ActivationTest")
{
    const int testLoops = 3;
    SUBCASE("ReLU")
    {
        std::cout << "ReLU Test" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithOneArgumentNormal(false, 1.0f, Compute::ReLU, 0, 0.1f);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("SoftMax")
    {
        std::cout << "SoftMax Test" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TestWithOneArgumentNormal(false, 1.0f, Compute::SoftMax, 0, 1.0f);
        Util::ResourceManager::ClearAll();
    }
}
#endif

#ifdef GemmTest
TEST_CASE("Gemm Test")
{
    const int testLoops = 3;
    SUBCASE("Gemm With Cuda")
    {
        for (int loopIdx = 0; loopIdx < testLoops; loopIdx++)
        {
            std::cout << "Gemm test : " << loopIdx << std::endl;
            Gemm1(false);
        }
        Util::ResourceManager::ClearAll();
    }
}
#endif

#ifdef GemmBroadcastTest
TEST_CASE("Gemm Broadcast Test")
{
    const int testLoops = 3;
    SUBCASE("Broadcast test with 1 dimension")
    {
        for (int i = 0; i < testLoops; i++)
            BroadcastWithOneDimension(false);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Broadcast test with Missing dimension")
    {
        for (int i = 0; i < testLoops; i++)
            BroadcastWithMissingDimension(false);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Broadcast test mixed")
    {
        for (int i = 0; i < testLoops; i++)
            BroadcastMixed(false);
        Util::ResourceManager::ClearAll();
    }
}
#endif

#ifdef InitializeTest
TEST_CASE("InitializeTest")
{
    const int testLoops = 3;
    SUBCASE("Initialize Ones")
    {
        std::cout << "Initialize Ones" << std::endl;
        for (int i = 0; i < testLoops; i++)
        {
            EqualInitializeTest(Compute::Initialize::Ones, false);
        }
    }
    SUBCASE("Initialize Normal")
    {
        std::cout << "Initialize Normal" << std::endl;
        for (int i = 0; i < testLoops; i++)
        {
            NoneZeroTest(Compute::Initialize::Normal, false, 100.0f, 1.0f);
        }
    }
}
#endif

#ifdef ConvolutionTest
TEST_CASE("Convolution")
{
    const int testLoops = 3;
    SUBCASE("Im2ColHost")
    {
        std::cout << "Im2Col && Col2Im" << std::endl;
        HostIm2ColTest(false);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("HostConv2D")
    {
        std::cout << "Host Conv2D" << std::endl;
        HostConv2DTest(false);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("Conv2D")
    {
        std::cout << "Conv2D" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            Conv2DTest(false, false);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("MaxPool2D")
    {
        std::cout << "MaxPool2D" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            MaxPool2DTest(false, false);
        Util::ResourceManager::ClearAll();
    }

    SUBCASE("AvgPool2D")
    {
        std::cout << "AvgPool2D" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            AvgPool2DTest(false, false);
        Util::ResourceManager::ClearAll();
    }
}
#endif

#ifdef GraphTest
TEST_CASE("BasicGraphTest")
{
    SUBCASE("BasicGraph")
    {
        std::cout << "Basic graph test" << std::endl;
        GraphFunctionalityTest();
    }

    SUBCASE("MultiplyTest")
    {
        std::cout << "Multiply" << std::endl;
        TestMultiply(false);
    }

    SUBCASE("MeanTest")
    {
        std::cout << "Mean" << std::endl;
        TestMean(false);
    }

    SUBCASE("MSETest")
    {
        std::cout << "MSE" << std::endl;
        TestMSE(false);
    }

    SUBCASE("AddTest")
    {
        std::cout << "Add" << std::endl;
        TestAdd(false);
    }

    SUBCASE("Linear Test")
    {
        std::cout << "Linear" << std::endl;
        TestLinear(false);
    }

    SUBCASE("Conv2DTest")
    {
        std::cout << "Conv2D" << std::endl;
        for (int i = 0; i < 3; ++i)
            TestConv2D(false);
    }

    SUBCASE("SoftmaxTest")
    {
        std::cout << "Softmax" << std::endl;
        for (int i = 0; i < 1; ++i)
            TestSoftmax(true);
    }
}
#endif

#ifdef ModelTest

TEST_CASE("Model Test")
{
    SUBCASE("SimpleLinearModelTest")
    {
        int xFeatures = 300;
        int yFeatures = 300;
        int batchSize = 10;
        std::vector<float> xFeatureVector(xFeatures * batchSize, 0.1f);
        std::vector<float> labelVector(yFeatures * batchSize, 10.0f);

        std::cout << "--- Simple Linear Model ---" << std::endl;

        SimpleLinearModel(xFeatureVector, labelVector, xFeatures, yFeatures,
                          0.0001f, batchSize, 2000, false);
    }

    SUBCASE("Conv2DModelTest")
    {
        const auto xChannels = 3;
        const auto yChannels = 3;
        const auto batchSize = 1;
        const auto xSize = std::make_pair(5, 5);
        const auto filterSize = std::make_pair(3, 3);
        const auto stride = std::make_pair(2, 2);
        const auto padSize = std::make_pair(2, 2);
        const auto dilation = std::make_pair(1, 1);
        const auto learningRate = 0.001f;
        const auto hostMode = false;
        const auto epochs = 1000;

        const auto [xRows, xCols] = xSize;
        const auto [filterRows, filterCols] = filterSize;
        const auto [strideRows, strideCols] = stride;
        const auto [padRows, padCols] = padSize;
        const auto [dilationRows, dilationCols] = dilation;

        const auto yRows =
            (xRows + 2 * padRows - dilationRows * (filterRows - 1) - 1) /
            strideRows +
            1;
        const auto yCols =
            (xCols + 2 * padCols - dilationCols * (filterCols - 1) - 1) /
            strideCols +
            1;

        std::vector<float> xFeatureVector(
            batchSize * xChannels * xRows * xCols,
            0.1f);
        std::vector<float> labelVector(
            batchSize * yChannels * yRows * yCols
            , 10.0f);

        std::cout << "--- Simple Conv2D Model ---" << std::endl;
        Conv2DModel(xFeatureVector, labelVector, batchSize, yChannels,
                    xChannels, xSize,
                    std::make_pair(yRows, yCols),
                    filterSize, stride, padSize, dilation, learningRate,
                    hostMode, epochs);
    }
}

#endif
} // namespace Sapphire::Test
