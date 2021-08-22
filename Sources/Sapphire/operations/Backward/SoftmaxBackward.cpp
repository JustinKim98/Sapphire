// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Backward/SoftmaxBackward.hpp>
#include <Sapphire/compute/ActivationOps.hpp>

namespace Sapphire::BackProp
{
constexpr int dyIdx = 0;
constexpr int xIdx = 0;
constexpr int dxIdx = 0;

SoftMaxBackward::SoftMaxBackward(TensorUtil::TensorData dx,
                                 TensorUtil::TensorData dy,
                                 TensorUtil::TensorData x)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) }, { std::move(x) },
                      {})
{
}


void SoftMaxBackward::m_runBackProp()
{
    const auto& dy = m_dyVector[dyIdx];
    const auto& x = m_constants[xIdx];
    auto& dx = m_dxVector[dxIdx];

    Compute::SoftMaxBackward(dx, dy, x);
}
}
