// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/tensor/TensorDescriptor.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <algorithm>

namespace Sapphire::TensorUtil
{
TensorDescriptor::TensorDescriptor(const Shape& shape, Type type,
                                   const Device& device, unsigned int batchSize,
                                   int key)
    : ForwardData(shape, type, device, batchSize, key),
      m_key(key),
      m_batchSize(batchSize),
      m_trainable(false)
{
}

TensorDescriptor::TensorDescriptor(TensorDescriptor&& tensorData) noexcept
    : ForwardData(std::move(tensorData.ForwardData)),
      BackwardData(std::move(tensorData.BackwardData)),
      m_key(tensorData.m_key),
      m_batchSize(tensorData.m_batchSize),
      m_trainable(tensorData.m_trainable),
      m_history(std::move(tensorData.m_history))
{
}

TensorDescriptor& TensorDescriptor::operator=(
    TensorDescriptor&& tensorDesc) noexcept
{
    ForwardData = tensorDesc.ForwardData;
    BackwardData = tensorDesc.BackwardData;
    m_key = tensorDesc.m_key;
    m_batchSize = tensorDesc.m_batchSize;
    m_trainable = tensorDesc.m_trainable;
    m_history = std::move(tensorDesc.m_history);
    return *this;
}

void TensorDescriptor::AppendOutputHistory(
    std::unique_ptr<BackProp::BackPropWrapper> wrapper, bool saveOutput)
{
    m_history.emplace_back(History(std::move(wrapper)));
}

void TensorDescriptor::AppendOperandHistory(int tensorDescKey)
{
    if (m_history.empty() || m_history.back().IsOutput)
    {
        History history;
        history.AddGradientInputTensorDescKey(tensorDescKey);
        m_history.emplace_back(std::move(history));
        return;
    }

    m_history.back().AddGradientInputTensorDescKey(tensorDescKey);
}

void TensorDescriptor::RemoveGradientInput(int tensorDescKey)
{
    if (m_history.empty() || m_history.back().IsOutput)
    {
        throw std::runtime_error(
            "RemoveGradientInput - Last history was empty or output");
    }

    auto& history = m_history.back();
    const auto it = std::find(history.GradientInputTensorKeyList.begin(),
                              history.GradientInputTensorKeyList.end(),
                              tensorDescKey);

    if (it == history.GradientInputTensorKeyList.end())
    {
        throw std::runtime_error(
            "RemoveGradientInput - tensorDescKey not found in gradient input "
            "tensor key "
            "list");
    }

    history.GradientInputTensorKeyList.erase(it);
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
    if (m_history.back().IsOutput)
        return true;

    if (const auto& lastHistory = m_history.back();
        lastHistory.GradientInputTensorKeyList.empty())
        return true;

    return false;
}
} // namespace Sapphire::TensorUtil
