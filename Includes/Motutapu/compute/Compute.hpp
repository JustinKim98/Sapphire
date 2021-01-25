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

void Add(Util::TensorData& out, const Util::TensorData& add);
//! Performs out = a + b

void Add(Util::TensorData& out, const Util::TensorData& a,
         const Util::TensorData& b);

//! Performs out = out - sub

void Sub(Util::TensorData& out, const Util::TensorData& sub);

//! Performs out = a - b

void Sub(Util::TensorData& out, const Util::TensorData& a,
         const Util::TensorData& b);

//! Performs out = a * b

void Mul(Util::TensorData& out, const Util::TensorData& a,
         const Util::TensorData& b);

//! Performs out = out * a

void Mul(Util::TensorData& out, const Util::TensorData& a);

//! Performs GEMM (out = a*b + c)

void Gemm(Util::TensorData& out, const Util::TensorData& a,
          const Util::TensorData& b, const Util::TensorData& c);

//! Performs in place GEMM (out = a*b + out)

void Gemm(Util::TensorData& out, const Util::TensorData& a,
          const Util::TensorData& b);

//! Performs out = out * factor

void Scale(Util::TensorData& out, float factor);

//! Performs out = a*factor

void Scale(Util::TensorData& out, const Util::TensorData& a, float factor);

//! Performs out = transpose(out)

void Transpose(Util::TensorData& out);

//! Performs out = transpose(a)

void Transpose(Util::TensorData& out, const Util::TensorData& a);

//! Performs out = out^factor for each element

void Pow(Util::TensorData& out, int factor);

//! Performs out = a^factor for each element

void Pow(Util::TensorData& out, const Util::TensorData& a);
}

#endif
