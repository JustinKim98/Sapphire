// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/tensor/TensorDescriptor.hpp>
#include <Motutapu/util/MemoryManager.hpp>
#include <algorithm>

namespace Motutapu::Util
{
TensorDescriptor::TensorDescriptor(const Shape &shape, Type type,
                                   const Device &device, unsigned int batchSize)
    : ForwardData(shape, type, device, batchSize),
      m_requireOutputSaving(false),
      m_trainable(false)
{
}

TensorDescriptor::TensorDescriptor(const Shape &shape, Type type,
                                   const Device &device, unsigned int batchSize,
                                   bool requireOutputSaving)
    : ForwardData(shape, type, device, batchSize),
      BackwardData(shape, type, device, batchSize),
      m_requireOutputSaving(requireOutputSaving),
      m_trainable(true)
{
}

TensorDescriptor::TensorDescriptor(TensorDescriptor &&tensorData) noexcept
    : ForwardData(tensorData.ForwardData),
      BackwardData(tensorData.BackwardData),
      m_requireOutputSaving(tensorData.m_requireOutputSaving),
      m_trainable(tensorData.m_trainable),
      m_history(std::move(tensorData.m_history))
{
}

TensorDescriptor &TensorDescriptor::operator=(
    TensorDescriptor &&tensorData) noexcept
{
    ForwardData = tensorData.ForwardData;
    BackwardData = tensorData.BackwardData;
    m_requireOutputSaving = tensorData.m_requireOutputSaving;
    m_trainable = tensorData.m_trainable;

    m_history = std::move(tensorData.m_history);
    return *this;
}

void TensorDescriptor::AppendOutputHistory(
    std::unique_ptr<BackProp::BackPropWrapper> wrapper, bool saveOutput)
{
    m_requireOutputSaving = saveOutput;
    m_history.emplace_back(History(std::move(wrapper)));
}

void TensorDescriptor::AppendOperandHistory(unsigned int tensorKey)
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

    auto &history = m_history.back();
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

void TensorDescriptor::PopIfOperandHistory()
{
    if (!m_history.empty() && !m_history.back().IsOutput)
        m_history.pop_back();
}

void TensorDescriptor::PopHistory()
{
    if (!m_history.empty())
        m_history.pop_back();
}

bool TensorDescriptor::IsBackPropReady() const

{
    if (m_history.empty())
        return false;
    else if (m_history.back().IsOutput)
        return true;
    else
    {
        const auto &lastHistory = m_history.back();
        if (lastHistory.GradientInputTensorKeys.empty())
            return true;
    }

    return false;
}
}  // namespace Motutapu::Util
