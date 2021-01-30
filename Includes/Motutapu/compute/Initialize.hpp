// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_INITIALIZE_HPP
#define MOTUTAPU_INITIALIZE_HPP

#include <Motutapu/tensor/TensorData.hpp>

namespace Motutapu::Compute
{
void NormalFloat(Util::TensorData data, float mean, float sd,
                        unsigned int size);

void Ones(Util::TensorData data);

void Zeros(Util::TensorData data);

void HeNormal(Util::TensorData data, int fanIn);

void Xavier(Util::TensorData data, int fanIn, int fanOut);

}  // namespace Motutapu::Compute

#endif  // MOTUTAPU_INITIALIZE_HPP
