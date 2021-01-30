// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_FUNCTIONAL_MATHFORWARD_HPP
#define MOTUTAPU_FUNCTIONAL_MATHFORWARD_HPP

#include <Motutapu/tensor/Tensor.hpp>

namespace Motutapu::NN::Functional
{
[[maybe_unused]]
static Tensor MulOp(const Tensor& a, const Tensor& b);

[[maybe_unused]]
static Tensor AddOp(const Tensor& a, const Tensor& b);

[[maybe_unused]]
static void AddOpInplace(const Tensor& out, Tensor& a);
}

#endif
