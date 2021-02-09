// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <Motutapu/Tests/BroadcastTest.hpp>
#include <Motutapu/Tests/ComputationTest.hpp>
#include <Motutapu/Tests/CudaFunctionalityTest.cuh>
#include <Motutapu/Tests/Test.hpp>
#include <iostream>
#include "doctest.h"

namespace Motutapu::Test
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

TEST_CASE("Computation Test")
{
    const int testLoops = 3;
    SUBCASE("Gemm With Cuda")
    {
        for (int loopIdx = 0; loopIdx < testLoops; loopIdx++)
        {
            std::cout << "Gemm test 1 : " << loopIdx << std::endl;
            TestGemm1();
        }
    }

    SUBCASE("Initialize test With Cuda")
    {
        for (int loopIdx = 0; loopIdx < testLoops; loopIdx++)
        {
            std::cout << "Gemm test 2 : " << loopIdx << std::endl;
            TestGemm2();
        }
    }

    SUBCASE("Gemm Broadcast")
    {
        for (int loopIdx = 0; loopIdx < testLoops; loopIdx++)
        {
            std::cout << "Gemm test Broadcast : " << loopIdx << std::endl;
            TestGemmBroadcast();
        }
    }
}

TEST_CASE("Broadcast Test")
{
    const int testLoops = 3;
    SUBCASE("Broadcast test with 1 dimension")
    {
        for (int i = 0; i < testLoops; i++)
            BroadcastWithOneDimension();
    }

    SUBCASE("Broadcast test with Missing dimension")
    {
        for (int i = 0; i < testLoops; i++)
            BroadcastWithMissingDimension();
    }

    SUBCASE("Broadcast test mixed")
    {
        for (int i = 0; i < testLoops; i++)
            BroadcastMixed();
    }
}

}  // namespace Motutapu::Test
