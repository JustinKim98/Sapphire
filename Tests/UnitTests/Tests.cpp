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
#include <OperationTest/MaxPool2DTest.hpp>
#include <OperationTest/CrossEntropyTest.hpp>
#include <ModelTest/Conv2DModel.hpp>
#include <ModelTest/MnistLinear.hpp>
#include <BasicsTest/TransposeTest.hpp>
#include <TensorTest/TensorFunctionalityTest.hpp>
#include <DataLoaderTest/CsvLoaderTest.hpp>
#include <FunctionTest/Conv2DTest.hpp>
#include <Sapphire/compute/TrigonometricOps.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/ActivationOps.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <TestUtil.hpp>
#include <iostream>
#include "doctest.h"

#define GraphTest
// #define DataLoaderTest
#define TrainTest
#define TensorFunctionalityTest
#define BasicsTest
#define ActivationTest
#define GemmTest
#define GemmBroadcastTest
#define InitializeTest
#define ConvolutionTest
#define BasicGraphTest
// #define ModelTest

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
    constexpr int testLoops = 3;
    SUBCASE("Transpose")
    {
        std::cout << "Transpose" << std::endl;
        for (int i = 0; i < testLoops; ++i)
            TransposeTest(false);
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
    constexpr int testLoops = 3;
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
    constexpr int testLoops = 3;
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
    constexpr int testLoops = 3;
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
    SUBCASE("Initialize Ones")
    {
        std::cout << "Initialize Ones" << std::endl;
        EqualInitializeTest(Compute::Initialize::Ones, false);
    }
    SUBCASE("Initialize Normal")
    {
        std::cout << "Initialize Normal" << std::endl;
        NoneZeroTest(Compute::Initialize::Normal, false, 100.0f, 1.0f);
    }
}
#endif

#ifdef ConvolutionTest
TEST_CASE("Convolution")
{
    SUBCASE("Im2ColHost")
    {
        std::cout << "Testing Im2Col && Col2Im ...";
        HostIm2ColTest(false);
        Util::ResourceManager::ClearAll();
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("HostConv2D")
    {
        std::cout << "Testing Conv2D on Host ... ";
        HostConv2DTest(false);
        Util::ResourceManager::ClearAll();
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("Cuda Conv2D")
    {
        std::cout << "Testing Conv2D on Cuda ... ";
        CudaConv2DTest(false, false);
        Util::ResourceManager::ClearAll();
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("MaxPool2D")
    {
        std::cout << "Testing MaxPool2D ... ";
        MaxPool2DTest(false, false);
        Util::ResourceManager::ClearAll();
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("AvgPool2D")
    {
        std::cout << "Testing AvgPool2D ... ";
        AvgPool2DTest(false, false);
        Util::ResourceManager::ClearAll();
        std::cout << "Done!" << std::endl;
    }
}
#endif

#ifdef GraphTest
TEST_CASE("BasicGraphTest")
{
    SUBCASE("MultiplyTest")
    {
        std::cout << "Testing MatMul ... ";
        TestMatMul(false);
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("MeanTest")
    {
        std::cout << "Testing Mean ... ";
        TestMean(false);
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("MSETest")
    {
        std::cout << "Testing MSE ... ";
        TestMSE(false);
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("CrossEntropyTest")
    {
        std::cout << "Testing CrossEntropy ... ";
        TestCrossEntropy(false);
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("AddTest")
    {
        std::cout << "Testing Add ... ";
        TestAdd(false);
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("SubTest")
    {
        std::cout << "Testing Sub ... ";
        TestAdd(false);
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("DotTest")
    {
        std::cout << "Testing Dot ... ";
        TestAdd(false);
        std::cout << "Done!" << std::endl;
    }

    SUBCASE("Linear Test")
    {
        std::cout << "Linear" << std::endl;
        TestLinear(false);
    }

    SUBCASE("CudaConv2DTest")
    {
        std::cout << "Conv2D" << std::endl;
        TestConv2D(false);
    }

    SUBCASE("MaxPool2DTest")
    {
        std::cout << "MaxPool2D" << std::endl;
        TestMaxPool2D(false);
    }

    SUBCASE("SoftmaxTest")
    {
        std::cout << "Softmax" << std::endl;
        TestSoftmax(false);
    }
}
#endif

#ifdef DataLoaderTest
TEST_CASE("Data Loader Test")
{
    SUBCASE("Csv Loader Test")
    {
#ifdef _MSC_VER
        const std::string filePath =
            "C:\\Users\\user\\Documents\\Sapphire\\Datasets\\train.csv";
#else
        const std::string filePath =
            "/mnt/c/Users/user/Documents/Sapphire/Datasets/train.csv";
#endif
        std::cout << "Testing csv loader " << std::endl;
        CsvLoaderTest(filePath, false);
    }
}
#endif

#ifdef TrainTest
TEST_CASE("train test")
{
    SUBCASE("Linear Train")
    {
        std::cout << "Testing Linear training with MSE" << std::endl;
        TestLinearTraining(false);
    }

    SUBCASE("Conv2D Train")
    {
        std::cout << "Testing Conv2D training with MSE" << std::endl;
        TestConv2DTraining(false);
    }

    SUBCASE("CrossEntropy Train")
    {
        std::cout << "Testing Cross Entropy training" << std::endl;
        TestCrossEntropyTraining(false);
    }
}
#endif

#ifdef ModelTest

TEST_CASE("Model Test")
{
    SUBCASE("MnistTest")
    {
        std::cout << "--- Mnist linear model ---" << std::endl;
#ifdef _MSC_VER
        const std::string filePath =
            "C:\\Users\\user\\Documents\\Sapphire\\Datasets\\train.csv";
#else
        const std::string filePath =
            "/mnt/c/Users/user/Documents/Sapphire/Datasets/train.csv";
#endif

        MnistLinear(
            filePath, 100,
            0.0001f, 3000, false);
    }

    SUBCASE("Conv2DModelTest")
    {
        constexpr auto batchSize = 100;

#ifdef _MSC_VER
        const std::string filePath =
            "C:\\Users\\user\\Documents\\Sapphire\\Datasets\\cifar-10-batches-"
            "bin\\data_batch_1.bin";
#else
        const std::string filePath =
            "/mnt/c/Users/user/Documents/Sapphire/Datasets/"
            "cifar-10-batches-bin/"
            "data_batch_1.bin";
#endif

        std::cout << "--- Cifar-10 Conv2D image classification model ---" <<
            std::endl;
        Conv2DModelTest(filePath,
                        batchSize, 0.002f, false, 1000);
    }
}

#endif
} // namespace Sapphire::Test
