// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_ACTIVATION_OPS_CUH
#define SAPPHIRE_COMPUTE_ACTIVATION_OPS_CUH

#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::Compute
{
using namespace TensorUtil;

void ReLU(TensorData& y, const TensorData& x);

void LeakyReLU(TensorData& y, const TensorData& x, float a);

void SoftMax(TensorData& y, const TensorData& x);

void ReLUBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void LeakyReLUBackward(TensorData& dx, const TensorData& dy, const TensorData& x, float a);

void SoftMaxBackward(TensorData& dx, const TensorData& dy, const TensorData& y);

}

#endif
