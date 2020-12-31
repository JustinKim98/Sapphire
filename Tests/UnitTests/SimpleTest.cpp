#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <Motutapu/Test.hpp>

TEST_CASE("Simple test")
{
    CHECK(Add(2, 3) == 5);
}

#ifdef WITH_CUDA
TEST_CASE("Check cuda")
{
    PrintCudaVersion();
}
#endif
