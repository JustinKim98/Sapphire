// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.
#ifndef MOTUTAPU_MSE_HPP
#define MOTUTAPU_MSE_HPP

#include <Motutapu/tensor/Tensor.hpp>

namespace Motutapu::NN::Loss
{
static Tensor MSE(const Tensor& x, const Tensor& label);
}

#endif  // MOTUTAPU_MSE_HPP