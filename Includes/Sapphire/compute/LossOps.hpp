// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_LOSS_OPS_HPP
#define SAPPHIRE_COMPUTE_LOSS_OPS_HPP

#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::Compute
{
void CrossEntropy(TensorUtil::TensorData& y, const TensorUtil::TensorData& x,
                  const TensorUtil::TensorData& label);

void CrossEntropyBackward(TensorUtil::TensorData& dx,
                          const TensorUtil::TensorData& label);
}

#endif
