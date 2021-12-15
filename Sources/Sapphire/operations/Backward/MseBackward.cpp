// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/MSEBackward.hpp>

namespace Sapphire::BackProp
{
constexpr int xIdx = 0;
constexpr int labelIdx = 1;
constexpr int dxIdx = 0;

MSEBackward::MSEBackward(std::string name, TensorUtil::TensorData dx,
                         TensorUtil::TensorData x,
                         TensorUtil::TensorData label)
    : BackPropWrapper(std::move(name), { std::move(dx) },
                      { TensorUtil::TensorData() },
                      { std::move(x), std::move(label) },
                      {})

{
}

void MSEBackward::m_runBackProp()
{
    auto x = m_constants[xIdx];
    auto label = m_constants[labelIdx];
    auto dx = m_dxVector[dxIdx];
    TensorUtil::TensorData diff(label.GetShape(), label.GetType(),
                                label.GetCudaDevice(), false);
    diff.SetMode(label.Mode());

    Compute::Sub(diff, x, label);
    Compute::Scale(dx, diff, 2.0f / static_cast<float>(dx.GetShape().At(-1)));
}
} // namespace Sapphire::BackProp
