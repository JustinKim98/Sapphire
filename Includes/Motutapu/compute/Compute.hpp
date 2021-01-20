// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_HPP
#define MOTUTAPU_COMPUTE_HPP

#include <Motutapu/tensor/TensorData.hpp>

namespace Motutapu::Compute
{
//! Performs out = out + add
template <typename T>
void Add(Util::TensorData<T>& out, const Util::TensorData<T>& add);

//! Performs out = a + b
template <typename T>
void Add(Util::TensorData<T>& out, const Util::TensorData<T>& a,
         const Util::TensorData<T>& b);

//! Performs out = out - sub
template <typename T>
void Sub(Util::TensorData<T>& out, const Util::TensorData<T>& sub);

//! Performs out = a - b
template <typename T>
void Sub(Util::TensorData<T>& out, const Util::TensorData<T>& a,
         const Util::TensorData<T>& b);

//! Performs out = a * b
template <typename T>
void Mul(Util::TensorData<T>& out, const Util::TensorData<T>& a,
         const Util::TensorData<T>& b);

//! Performs out = out * a
template <typename T>
void Mul(Util::TensorData<T>& out, const Util::TensorData<T>& a);

//! Performs GEMM (out = a*b + c)
template <typename T>
void Gemm(Util::TensorData<T>& out, const Util::TensorData<T>& a,
          const Util::TensorData<T>& b, const Util::TensorData<T>& c);

//! Performs in place GEMM (out = a*b + out)
template <typename T>
void Gemm(Util::TensorData<T>& out, const Util::TensorData<T>& a,
          const Util::TensorData<T>& b);

//! Performs out = out * factor
template <typename T>
void Scale(Util::TensorData<T>& out, T factor);

//! Performs out = a*factor
template <typename T>
void Scale(Util::TensorData<T>& out, const Util::TensorData<T>& a, T factor);

//! Performs out = transpose(out)
template <typename T>
void Transpose(Util::TensorData<T>& out);

//! Performs out = transpose(a)
template <typename T>
void Transpose(Util::TensorData<T>& out, const Util::TensorData<T>& a);

//! Performs out = out^factor for each element
template <typename T>
void Pow(Util::TensorData<T>& out, int factor);

//! Performs out = a^factor for each element
template <typename T>
void Pow(Util::TensorData<T>& out, const Util::TensorData<T>& a);


}

#endif
