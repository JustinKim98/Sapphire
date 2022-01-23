// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_CREATE_TENSOR_HPP
#define SAPPHIRE_CREATE_TENSOR_HPP


#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>

namespace Sapphire
{
Tensor MakeTensor(const Shape& shape,
                  std::unique_ptr<Initialize::Initializer> initializer,
                  bool preserve);

Tensor MakeTensor(const Shape& shape, const DeviceInfo& device,
                  std::unique_ptr<Initialize::Initializer> initializer,
                  bool preserve = false);
}

#endif
