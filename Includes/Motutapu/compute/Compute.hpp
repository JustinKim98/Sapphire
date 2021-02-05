// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_COMPUTE_DECL_HPP
#define MOTUTAPU_COMPUTE_COMPUTE_DECL_HPP

#include <Motutapu/tensor/TensorData.hpp>

namespace Motutapu::Compute
{
//! Performs out = out + add

using namespace TensorUtil;

void Add(TensorData& out, const TensorData& add);

//! Performs out = a + b
void Add(TensorData& out, const TensorData& a, const TensorData& b);

//! Performs out = out - sub
void Sub(TensorData& out, const TensorData& sub);

//! Performs out = a - b

void Sub(TensorData& out, const TensorData& a, const TensorData& b);

//! Performs out = a * b + out
void Mul(TensorData& out, const TensorData& a, const TensorData& b);

//! Performs GEMM (out = a*b + c)
[[maybe_unused]] void Gemm(TensorData& out, const TensorData& a,
                           const TensorData& b, const TensorData& c);

//! Performs out = a*factor
void Scale(TensorData& out, const TensorData& a, float factor);

//! Performs out = transposeKernel(a)
void Transpose(TensorData& out, const TensorData& a);

//! Performs out = out^factor for each element
void Pow(TensorData& out, int factor);

//! Performs out = a^factor for each element
void Pow(TensorData& out, const TensorData& a, int factor);
}  // namespace Motutapu::Compute

#endif
