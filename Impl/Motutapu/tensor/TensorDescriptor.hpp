// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_TENSORDESCRIPTOR_HPP
#define MOTUTAPU_UTIL_TENSORDESCRIPTOR_HPP

#include <Motutapu/tensor/TensorDescriptorDecl.hpp>
#include <algorithm>

namespace Motutapu::Util
{
template <typename T>
void TensorDescriptor<T>::AppendOutputHistory(
    std::unique_ptr<BackProp::BackPropWrapper<T>> wrapper,
    bool saveOutput)
{
    m_requireOutputSaving = saveOutput;
    m_history.emplace_back(History(std::move(wrapper)));
}

template <typename T>
void TensorDescriptor<T>::AppendOperandHistory(int tensorKey)
{
    if (m_history.empty() || m_history.back().IsOutput)
    {
        History history;
        history.AddGradientInputTensorKey(tensorKey);
        m_history.emplace_back(history);
        return;
    }

    auto history&  = m_history.back();
    history.AddGradientInputTensorKey(tensorKey);
}

template <typename T>
void TensorDescriptor<T>::RemoveGradientInputKey(int tensorKey)
{
    if (m_history.empty() || m_history.back().IsOutput)
    {
        throw std::runtime_error(
            "RemoveGradientInputKey - Last history was empty or output");
    }

    auto& history = m_history.back();
    auto it = std::find(history.GradientInputTensorKeys.begin(),
                        history.GradientInputTensorKeys.end(), tensorKey);

    if (it == history.GradientInputTensorKeys.end())
    {
        throw std::runtime_error(
            "RemoveGradientInputKey - tensorKey not found in gradient input tensor key "
            "list");
    }

    history.GradientInputTensorKeys.erase(it);
}

template <typename T>
void TensorDescriptor<T>::PopHistory()
{
    if (!m_history.empty())
        m_history.pop_back();
}
}

#endif
