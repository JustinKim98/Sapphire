// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.
#include <Sapphire/operations/Backward/Indexing/FlattenBackward.hpp>
#include <Sapphire/compute/IndexingOps.hpp>

namespace Sapphire::BackProp
{
FlattenBackward::FlattenBackward(TensorUtil::TensorData dx,
                                 TensorUtil::TensorData dy, Shape shape)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) }),
      m_shape(std::move(shape))
{
}

void FlattenBackward::m_runBackProp()
{
    auto dx = m_dxVector[0];
    auto dy = m_dyVector[0];
    Compute::Reshape(m_dxVector[0], m_shape);
}
}
