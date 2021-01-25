// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/tensor/TensorDescriptor.hpp>
#include <algorithm>

namespace Motutapu::Util
{
void TensorDescriptor::AppendOutputHistory(
    std::unique_ptr<BackProp::BackPropWrapper> wrapper, bool saveOutput)
{
    m_requireOutputSaving = saveOutput;
    m_history.emplace_back(History(std::move(wrapper)));
}

void TensorDescriptor::AppendOperandHistory(int tensorKey)
{
    if (m_history.empty() || m_history.back().IsOutput)
    {
        History history;
        history.AddGradientInputTensorKey(tensorKey);
        m_history.emplace_back(std::move(history));
        return;
    }

    m_history.back().AddGradientInputTensorKey(tensorKey);
}

void TensorDescriptor::RemoveGradientInputKey(int tensorKey)
{
    if (m_history.empty() || m_history.back().IsOutput)
    {
        throw std::runtime_error(
            "RemoveGradientInputKey - Last history was empty or output");
    }

    auto& history = m_history.back();
    const auto it = std::find(history.GradientInputTensorKeys.begin(),
                        history.GradientInputTensorKeys.end(), tensorKey);

    if (it == history.GradientInputTensorKeys.end())
    {
        throw std::runtime_error(
            "RemoveGradientInputKey - tensorKey not found in gradient input "
            "tensor key "
            "list");
    }

    history.GradientInputTensorKeys.erase(it);
}

void TensorDescriptor::PopHistory()
{
    if (!m_history.empty())
        m_history.pop_back();
}
} // namespace Motutapu::Util
