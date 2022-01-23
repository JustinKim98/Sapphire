// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/tensor/CreateTensor.hpp>

namespace Sapphire
{
Tensor MakeTensor(const Shape& shape,
                  std::unique_ptr<Initialize::Initializer> initializer,
                  bool preserve)
{
    auto tensor = Tensor(shape, preserve);
    Initialize::Initialize(tensor, std::move(initializer));
    return tensor;
}

Tensor MakeTensor(const Shape& shape, const DeviceInfo& device,
                  std::unique_ptr<Initialize::Initializer> initializer,
                  bool preserve)
{
    auto tensor = Tensor(shape, device, preserve);
    Initialize::Initialize(tensor, std::move(initializer));
    return tensor;
}
}
