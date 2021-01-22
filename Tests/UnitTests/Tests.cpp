// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "ComputationTest/TestCudaGemm.hpp"
#include <Motutapu/Test.hpp>

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

TEST_CASE("GPU GEMM with tensor cores")
{
#ifdef WITH_CUDA
    TensorGemmTest();
#endif
}

TEST_CASE("GPU GEMM")
{
#ifdef WITH_CUDA
    FloatGemmTest();
#endif
}
}
