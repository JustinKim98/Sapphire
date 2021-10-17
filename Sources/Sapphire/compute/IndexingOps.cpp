// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/IndexingOps.hpp>

namespace Sapphire::Compute
{
using namespace TensorUtil;

void Flatten(TensorData& tensorData)
{
    const auto shape = tensorData.GetShape();
    const Shape newShape({ 1, shape.Size() });
    tensorData.Reshape(newShape);
}
}
