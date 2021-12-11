// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Backward/SoftmaxBackward.hpp>
#include <Sapphire/compute/ActivationOps.hpp>
#include <Sapphire/compute/Initialize.hpp>

namespace Sapphire::BackProp
{
constexpr int dyIdx = 0;
constexpr int yIdx = 0;
constexpr int dxIdx = 0;

SoftMaxBackward::SoftMaxBackward(std::string name, TensorUtil::TensorData dx,
                                 TensorUtil::TensorData dy,
                                 TensorUtil::TensorData y)
    : BackPropWrapper(std::move(name), { std::move(dx) }, { std::move(dy) },
                      { std::move(y) },
                      {})
{
}


void SoftMaxBackward::m_runBackProp()
{
    const auto& dy = m_dyVector[dyIdx];
    const auto& y = m_constants[yIdx];
    auto& dx = m_dxVector[dxIdx];

    Compute::SoftMaxBackward(dx, dy, y);
}
}
