// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_FUNCTIONAL_MATHFORWARD_HPP
#define Sapphire_FUNCTIONAL_MATHFORWARD_HPP

#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire::NN::Functional
{
[[maybe_unused]]
static Tensor MulOp(const Tensor& a, const Tensor& b);

[[maybe_unused]]
static Tensor AddOp(const Tensor& a, const Tensor& b);

[[maybe_unused]]
static void AddOpInplace(const Tensor& out, Tensor& a);
}

#endif
