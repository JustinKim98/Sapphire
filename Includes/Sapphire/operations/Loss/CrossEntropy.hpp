// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_NN_LOSS_CROSS_ENTROPY_HPP
#define SAPPHIRE_NN_LOSS_CROSS_ENTROPY_HPP

#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire::NN::Loss
{
[[maybe_unused]] Tensor CrossEntropy(const Tensor& input, const Tensor& label);
}


#endif  // !SAPPHIRE_NN_LOSS_CROSS_ENTROPY_HPP
