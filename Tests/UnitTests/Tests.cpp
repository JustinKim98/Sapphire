// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <Sapphire/Tests/BroadcastTest.hpp>
#include <Sapphire/Tests/GemmTest.hpp>
#include <Sapphire/Tests/CudaFunctionalityTest.cuh>
#include <Sapphire/Tests/SparseGemmTest.hpp>
#include <Sapphire/Tests/SparseMemoryTest.hpp>
#include <Sapphire/Tests/Test.hpp>
#include <Sapphire/Tests/OperationTest/LinearTest.hpp>
#include <Sapphire/Tests/Basics/TransposeTest.hpp>
#include <Sapphire/Tests/TensorTest/TensorFunctionalityTest.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <iostream>
#include "doctest.h"
#include <Sapphire/compute/BasicOps.hpp>

#include "Sapphire/compute/TrigonometricOps.hpp"
//#define EnableAllTest

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

TEST_CASE("Basics")
{
    SUBCASE("Transpose")
    {
        for (int i = 0; i < 5; ++i)
            TransposeTest(false);
    }

    SUBCASE("Add")
    {
        TestWithTwoArgumentsWithSameShape(false, 1.0f, Compute::Add);
    }

    SUBCASE("Sub")
    {
        TestWithTwoArgumentsWithSameShape(false, 1.0f, Compute::Sub);
    }

    SUBCASE("Dot")
    {
        TestWithTwoArgumentsWithSameShape(false, 1.0f, Compute::Dot);
    }

    SUBCASE("log")
    {
        TestWithOneArgument(false, 1.0f, Compute::log);
    }

    SUBCASE("Inverse")
    {
        TestWithOneArgument(false, 1.0f, Compute::Inverse);
    }

    SUBCASE("Sin")
    {
        TestWithOneArgument(false, 1.0f, Compute::Sin);
    }

    SUBCASE("Cos")
    {
        TestWithOneArgument(false, 1.0f, Compute::Cos);
    }

    SUBCASE("Tan")
    {
        TestWithOneArgument(false, 1.0f, Compute::Tan);
    }
}

TEST_CASE("Gemm Test")
{
    const int testLoops = 3;
    SUBCASE("Gemm With Cuda")
    {
        for (int loopIdx = 0; loopIdx < testLoops; loopIdx++)
        {
            std::cout << "Gemm test 1 : " << loopIdx << std::endl;
            Gemm1(true);
        }
    }
}

TEST_CASE("Gemm Broadcast Test")
{
    const int testLoops = 3;
    SUBCASE("Broadcast test with 1 dimension")
    {
        for (int i = 0; i < testLoops; i++)
            BroadcastWithOneDimension(true);
    }

    SUBCASE("Broadcast test with Missing dimension")
    {
        for (int i = 0; i < testLoops; i++)
            BroadcastWithMissingDimension(true);
    }

    SUBCASE("Broadcast test mixed")
    {
        for (int i = 0; i < testLoops; i++)
            BroadcastMixed(true);
    }
}

// TEST_CASE("GraphTest")
// {
//     SUBCASE("LinearForward Test")
//     {
//         Operation::LinearForwardTest();
//     }
// }

#ifdef EnableAllTest

TEST_CASE("SparseMemory function Test")
{
    SUBCASE("SparseMemoryAllocationHost")
    {
        std::cout << "Testing Sparse Memory Allocation for Host ...";
        SparseMemoryAllocationHost();
        std::cout << " Done" << std::endl;
    }

    SUBCASE("LoadDistMemoryAllocationHost")
    {
        std::cout << "Testing Load Distribution Memory Allocation forHost...";
        LoadDistMemoryAllocationHost();
        std::cout << " Done\n  " << std::endl;
    }

    SUBCASE("SparseMemoryDevice")
    {
        std::cout << "Testing Sparse Memory Allocation For Device ...";
        SparseMemoryAllocationDevice();
        std::cout << " Done" << std::endl;
    }

    SUBCASE("SparseMemoryCopy Device To Device")
    {
        std::cout << "Testing Sparse Memory Copy between device ...";
        SparseMemoryCopyDeviceToDevice();
        std::cout << " Done" << std::endl;
    }
}

TEST_CASE("Device Sparse Gemm Test")
{
    SUBCASE("Sparse Multiplication Test (complex)")
    {
        std::cout << "Testing Sparse Multiplication (complex) ..." << std::endl;
        const auto elapsedTime =
            SparseGemmTestComplex(50, 50, 50, 1, false, false);
        std::cout << " Done ... elapsed time (microSeconds) : " << elapsedTime
            << "\n"
            << std::endl;
    }

    SUBCASE("Sparse Multiplication Test (simple)")
    {
        std::cout << "Testing Sparse Multiplication (simple) ..." << std::endl;
        const auto elapsedTime = SparseGemmTestSimple(5, 5, 5, 5, false, false);
        std::cout << " Done ... elapsed time (microSeconds) : " << elapsedTime
            << "\n"
            << std::endl;
    }
}


TEST_CASE("Sparse Performance Test")
{
    SUBCASE("Matrix conversion test")
    {
        std::cout << "Testing conversion ..." << std::endl;
        SparseMatrixConversionTest(100, 100, 10, 0.1f, false);
        std::cout << " Done" << std::endl;
    }

    SUBCASE("Correctness test (Cuda)")
    {
        std::cout << "Testing correctness (Cuda) ..." << std::endl;
        SparseTestCorrectnessCuda(1000, 1000, 50, 3, 0.5f, false);
        SparseTestCorrectnessCuda(40, 50, 500, 3, 0.5f, false);
        std::cout << " Done" << std::endl;
    }

    SUBCASE("Correctness test (Host)")
    {
        std::cout << "Testing correctness (Host) ..." << std::endl;
        SparseTestCorrectnessHost(5, 5, 50, 3, 0.9f, false);
        SparseTestCorrectnessHost(500, 500, 500, 3, 0.5f, false);
        std::cout << " Done" << std::endl;
    }

    SUBCASE("General Performance Test")
    {
        const std::filesystem::path workDir = "/home/jwkim98/Desktop";
        const bool printResults = true;
        const bool writeResults = false;
        const size_t iterations = 10;

        std::cout << "Testing performance ..." << std::endl;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> uniform(100, 300);

        std::vector<PerformanceData> performanceData(10 * iterations);
        size_t count = 0;
        for (size_t i = 0; i < iterations; ++i)
        {
            float sparsity = 0.0f;
            while (sparsity < 1.0f)
            {
                performanceData[count] = PerformanceTest(
                    uniform(gen), uniform(gen), uniform(gen), 10, sparsity);
                if (printResults)
                    performanceData[count].PrintData();
                sparsity += 0.1f;
                count += 1;
            }
        }
        std::cout << " Done" << std::endl;

        if (writeResults)
        {
            std::filesystem::current_path(workDir);
            std::ofstream file;
            file.open("performance.csv", std::ios::out | std::ios::app);
            PerformanceData::WriteCsvHeader(file);
            for (const auto& data : performanceData)
            {
                data.WriteCsv(file);
            }
            file.close();
        }
    }
}
#endif
} // namespace Sapphire::Test
