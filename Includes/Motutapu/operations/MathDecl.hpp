// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_MATH_DECL_HPP
#define MOTUTAPU_MATH_DECL_HPP

#include <vector>

#include "UnitDecl.hpp"
#include "Motutapu/tensor/TensorDecl.hpp"

namespace Motutapu
{
template <typename T>
static std::vector<Tensor<T>> MulOp(std::vector<Tensor<T>>& inputs,
                                    Unit<T>& unit);
}


#endif
