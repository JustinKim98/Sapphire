// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Backward/CrossEntropyBackward.hpp>
#include <Sapphire/operations/Backward/BackPropWrapper.hpp>
#include <Sapphire/compute/LossOps.hpp>

namespace Sapphire::BackProp
{
constexpr int labelIdx = 0;
constexpr int dxIdx = 0;


CrossEntropyBackward::CrossEntropyBackward(TensorUtil::TensorData dx,
                                           TensorUtil::TensorData label)
    : BackPropWrapper({ std::move(dx) }, { TensorUtil::TensorData() },
                      { std::move(label) }, {})
{
}

void CrossEntropyBackward::m_runBackProp()
{
    const auto label = m_constants[labelIdx];
    auto dx = m_dxVector[dxIdx];

    Compute::CrossEntropyBackward(dx, label);
}
}
