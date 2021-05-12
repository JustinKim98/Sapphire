// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.
#ifndef Sapphire_MSE_HPP
#define Sapphire_MSE_HPP

#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire::NN::Loss
{
static Tensor MSE(const Tensor& x, const Tensor& label);
}

#endif  // Sapphire_MSE_HPP