// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/MSEBackward.hpp>

namespace Sapphire::BackProp
{
MSEBackward::MSEBackward(const TensorUtil::TensorData& x,
                         TensorUtil::TensorData dx,
                         const TensorUtil::TensorData& label,
                         TensorUtil::TensorData dy)
    : BackPropWrapper({ std::move(dy) }, { std::move(dx) })

{
    m_savedTensorMap["x"] = x.CreateCopy();
    m_savedTensorMap["label"] = label.CreateCopy();
}

bool MSEBackward::InvokeBackProp(const TensorUtil::TensorData& input)
{
    auto& x = m_savedTensorMap["x"];
    auto& label = m_savedTensorMap["label"];
    auto& dx = m_gradientOutputs[0];
    auto& dy = m_gradientInputs[0];

    Compute::Sub(dx, x, label);
    Compute::Scale(dx, dx, 2.0f / static_cast<float>(dy.TensorShape.Size()));
    Compute::Dot(dx, dx, dy);

    return true;
}
}  // namespace Sapphire::BackProp