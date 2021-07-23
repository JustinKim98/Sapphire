// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.
#ifndef SAPPHIRE_MSE_HPP
#define SAPPHIRE_MSE_HPP

#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire::NN::Loss
{
[[maybe_unused]] static Tensor MSE(const Tensor& input, const Tensor& label);
}

#endif  // Sapphire_MSE_HPP
