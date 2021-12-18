// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_SOFTMAX_HPP
#define SAPPHIRE_SOFTMAX_HPP

#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire::F
{
Tensor SoftMax(const Tensor& input);
}

#endif  // Sapphire_SOFTMAX_HPP
