// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire
{
Tensor::Tensor(const Shape& shape, const CudaDevice& device,
               Type type)
    : m_tensorDescKey(-1)
{
    auto& model = ModelManager::GetCurrentModel();
    m_tensorDescKey = model.RegisterTensorDescriptor(
        shape, type, device);
}

Tensor::Tensor(int descKey)
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

Shape Tensor::GetForwardDataShape() const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc =
        model.GetDescriptor(m_tensorDescKey);
    return desc.GetShape();
}

CudaDevice Tensor::GetDevice() const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(
        m_tensorDescKey);
    return desc.GetDevice();
}

int Tensor::TensorDescriptorKey() const
{
    return m_tensorDescKey;
}

void Tensor::ToCuda()
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(
        m_tensorDescKey);
    desc.ToCuda();
    desc.SetMode(DeviceType::Cuda);
}

void Tensor::ToHost()
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    desc.ToHost();
    desc.SetMode(DeviceType::Host);
}

DeviceType Tensor::Mode() const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    return desc.Mode();
}

void Tensor::SetMode(DeviceType mode)
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    desc.SetMode(mode);
}

const float* Tensor::GetRawData() const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    TensorUtil::TensorData tensorData = desc.GetForwardData();

    auto device = tensorData.GetCudaDevice();
    if (tensorData.GetCudaDevice().Type() == DeviceType::Cuda)
    {
        tensorData.SyncCudaDataWithHost();
    }

    return tensorData.GetDenseHost();
}
} // namespace Sapphire
