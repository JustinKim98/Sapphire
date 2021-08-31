// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Backward/ReLUBackward.hpp>
#include <Sapphire/compute/ActivationOps.hpp>

namespace Sapphire::BackProp
{
ReLUBackward::ReLUBackward(TensorUtil::TensorData dx, TensorUtil::TensorData dy,
                           TensorUtil::TensorData x)
    : BackPropWrapper(
        { std::move(dx) }, { std::move(dy) }, { std::move(x) }, {})
{
}

void ReLUBackward::m_runBackProp()
{
    auto x = m_constants[0];
    auto dx = m_dxVector[0];
    auto dy = m_dyVector[0];

    Compute::ReLUBackward(dx, dy, x);
}
}
