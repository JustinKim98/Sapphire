// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_NN_FUNCTIONAL_MAX_POOL_2D
#define SAPPHIRE_NN_FUNCTIONAL_MAX_POOL_2D

#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <utility>

namespace Sapphire::F
{
Tensor MaxPool2D(const Tensor& tensor, std::pair<int, int> windowSize,
                 std::pair<int, int> stride,
                 std::pair<int, int> padSize = std::pair(0, 0));
}

#endif
