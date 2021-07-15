// Copyright(c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include "doctest.h"
#include <random>
#include <cmath>
#include <Sapphire/tensor/Shape.hpp>

namespace Sapphire::Test
{
void Conv2DTestCuda()
{
    // std::random_device
    //     rd;  // Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(
    //     rd());  // Standard mersenne_twister_engine seeded with rd()
    // std::uniform_int_distribution<> distrib(1, 100);
    //
    // const unsigned int N = distrib(gen);
    // const unsigned int C = distrib(gen);
    // const unsigned int H = distrib(gen);
    // const unsigned int W = distrib(gen);
    //
    // const Shape InputShape({ N, C, H, W });
}
}
