// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_TRIGONOMETRIC_OPS_HPP
#define SAPPHIRE_COMPUTE_TRIGONOMETRIC_OPS_HPP
#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::Compute
{
using namespace TensorUtil;

void Cos(TensorData& y, const TensorData& x);

void Sin(TensorData& y, const TensorData& x);

void Tan(TensorData& y, const TensorData& x);

void Cosh(TensorData& y, const TensorData& x);

void Sinh(TensorData& y, const TensorData& x);

void Tanh(TensorData& y, const TensorData& x);

void ArcCos(TensorData& y, const TensorData& x);

void Arcsin(TensorData& y, const TensorData& x);

void ArcTan(TensorData& y, const TensorData& x);

void ArcCosh(TensorData& y, const TensorData& x);

void ArcSinh(TensorData& y, const TensorData& x);

void ArcTanh(TensorData& y, const TensorUtil::TensorData& x);
}

#endif
