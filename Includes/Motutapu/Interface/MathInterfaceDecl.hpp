// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_MATHINTERFACE_DECL_HPP
#define MOTUTAPU_MATHINTERFACE_DECL_HPP

#include <Motutapu/tensor/TensorDecl.hpp>

namespace Motutapu
{
template <typename T>
Tensor<T> abs(const Tensor<T>& tensor);

template <typename T>
void abs(Tensor<T>& tensor);

template <typename T>
Tensor<T> cos(const Tensor<T>& tensor);

template <typename T>
void cos(Tensor<T>& tensor);

template <typename T>
Tensor<T> sin(const Tensor<T>& tensor);

template <typename T>
void sin(Tensor<T>& tensor);

template <typename T>
Tensor<T> tan(const Tensor<T>& tensor);

template <typename T>
void tan(Tensor<T>& tensor);

template <typename T>
Tensor<T> acos(const Tensor<T>& tensor);

template <typename T>
void acos(Tensor<T>& tensor);

template <typename T>
Tensor<T> asin(const Tensor<T>& tensor);

template <typename T>
void asin(Tensor<T>& tensor);

template <typename T>
Tensor<T> atan(const Tensor<T>& tensor);

template <typename T>
void atan(Tensor<T>& tensor);

template <typename T>
Tensor<T> cosh(const Tensor<T>& tensor);

template <typename T>
void cosh(Tensor<T>& tensor);

template <typename T>
Tensor<T> sinh(const Tensor<T>& tensor);

template <typename T>
void sinh(Tensor<T>& tensor);

template <typename T>
Tensor<T> tanh(const Tensor<T>& tensor);

template <typename T>
void tanh(Tensor<T>& tensor);

template <typename T>
Tensor<T> exp(const Tensor<T>& tensor);

template <typename T>
void exp(Tensor<T>& tensor);

template <typename T>
Tensor<T> log(const Tensor<T>& tensor);

template <typename T>
void log(Tensor<T>& tensor);

template <typename T>
Tensor<T> log10(const Tensor<T>& tensor);

template <typename T>
void log10(Tensor<T>& tensor);

template <typename T>
Tensor<T> log2(const Tensor<T>& tensor);

template <typename T>
void log2(Tensor<T>& tensor);

template <typename T>
Tensor<T> neg(const Tensor<T>& tensor);

template <typename T>
void neg(Tensor<T>& tensor);

template <typename T>
Tensor<T> add(const Tensor<T>& tensorA, const Tensor<T>& tensorB);

template <typename T>
void add(Tensor<T>& tensor, const Tensor<T>& other);

template <typename T>
Tensor<T> sub(const Tensor<T>& tensorA, const Tensor<T>& tensorB);

template <typename T>
void sub(Tensor<T>& tensor, const Tensor<T>& other);

template <typename T>
Tensor<T> mul(const Tensor<T>& tensorA, const Tensor<T>& tensorB);

template <typename T>
void mul(Tensor<T>& tensor, const Tensor<T>& other);
}

#endif