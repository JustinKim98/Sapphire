// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_INITIALIZE_HPP
#define Sapphire_INITIALIZE_HPP

#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::Compute::Initialize
{
void Normal(TensorUtil::TensorData& data, float mean, float sd);

void Ones(TensorUtil::TensorData& data);

void Zeros(TensorUtil::TensorData& data);

void Scalar(TensorUtil::TensorData& data, float value);

void HeNormal(TensorUtil::TensorData& data, int fanIn);

void Xavier(TensorUtil::TensorData& data, int fanIn, int fanOut);
} // namespace Sapphire::Compute::Initialize

#endif  // Sapphire_INITIALIZE_HPP
