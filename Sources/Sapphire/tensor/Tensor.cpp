// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire
{
Tensor::Tensor(unsigned int descKey)
    : m_tensorDescKey(descKey)
{
}

Tensor& Tensor::operator=(const Tensor& tensor)
{
    if (&tensor == this)
        return *this;

    m_tensorDescKey = tensor.m_tensorDescKey;
    return *this;
}

Shape Tensor::GetShape() const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc =
        model.GetDescriptor(m_tensorDescKey);
    return desc.ForwardData.GetShape();
}

Device Tensor::GetDevice() const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(
        m_tensorDescKey);
    return desc.ForwardData.GetDevice();
}

int Tensor::TensorDescriptorKey() const
{
    return m_tensorDescKey;
}

void Tensor::SendTo(const Device& device) const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(
        m_tensorDescKey);
    desc.ForwardData.SendTo(device);
    if (desc.IsTrainable())
    {
        desc.BackwardData.SendTo(device);
    }
}
} // namespace Sapphire
