// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

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
#endif
}

TEST_CASE("Computation Test")
{
    SUBCASE("Basic functionality test")
    {
        MallocTest();
        std::cout << "Malloc test successful" << std::endl;
    }

    SUBCASE("Gemm With Cuda")
    {
        TestGemm();
        std::cout << "Gemm test successful" << std::endl;
    }
}

}  // namespace Motutapu::Test
