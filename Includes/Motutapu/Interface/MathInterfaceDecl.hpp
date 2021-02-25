// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_MATHINTERFACE_DECL_HPP
#define MOTUTAPU_MATHINTERFACE_DECL_HPP

#include <Motutapu/tensor/Tensor.hpp>

namespace Motutapu
{
Tensor abs(const Tensor& tensor);

void abs(Tensor& tensor);

Tensor cos(const Tensor& tensor);

void cos(Tensor& tensor);

Tensor sin(const Tensor& tensor);

void sin(Tensor& tensor);

Tensor tan(const Tensor& tensor);

void tan(Tensor& tensor);

Tensor acos(const Tensor& tensor);

void acos(Tensor& tensor);

Tensor asin(const Tensor& tensor);

void asin(Tensor& tensor);

Tensor atan(const Tensor& tensor);

void atan(Tensor& tensor);

Tensor cosh(const Tensor& tensor);

void cosh(Tensor& tensor);

Tensor sinh(const Tensor& tensor);

void sinh(Tensor& tensor);

Tensor tanh(const Tensor& tensor);

void tanh(Tensor& tensor);

Tensor exp(const Tensor& tensor);

void exp(Tensor& tensor);

Tensor log(const Tensor& tensor);

void log(Tensor& tensor);

Tensor log10(const Tensor& tensor);

void log10(Tensor& tensor);

Tensor log2(const Tensor& tensor);

void log2(Tensor& tensor);

Tensor neg(const Tensor& tensor);

void neg(Tensor& tensor);

Tensor add(const Tensor& tensorA, const Tensor& tensorB);

void add(Tensor& tensor, const Tensor& other);

Tensor sub(const Tensor& tensorA, const Tensor& tensorB);

void sub(Tensor& tensor, const Tensor& other);

Tensor mul(const Tensor& tensorA, const Tensor& tensorB);

void mul(Tensor& tensor, const Tensor& other);
}

#endif
