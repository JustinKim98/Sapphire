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

Shape Tensor::GetShape() const
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

std::unique_ptr<float[]> Tensor::GetForwardDataCopy() const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    const TensorUtil::TensorData tensorData = desc.GetForwardData();

    // if (desc.Mode() == DeviceType::Cuda)
    //     desc.ToHost();
    // else
    //     desc.ToCuda();

    auto tempPtr = std::unique_ptr<float[]>(
        new float[tensorData.GetShape().Size()]);
    std::size_t idx = 0;
    for (std::size_t ii = 0; ii < tensorData.DenseTotalLengthHost;
         ii += tensorData.PaddedHostColSize)
        for (std::size_t i = ii; i < ii + tensorData.GetShape().Cols(); ++i)
        {
            tempPtr[idx] = tensorData.GetDenseHost()[i];
            idx++;
        }

    return tempPtr;
}

std::unique_ptr<float[]> Tensor::GetBackwardDataCopy() const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    const TensorUtil::TensorData tensorData = desc.GetBackwardData();

    if (desc.Mode() == DeviceType::Cuda)
        desc.ToHost();
    else
        desc.ToCuda();

    auto tempPtr =
        std::unique_ptr<float[]>(new float[tensorData.GetShape().Size()]);
    std::size_t idx = 0;
    for (std::size_t ii = 0; ii < tensorData.DenseTotalLengthHost;
         ii += tensorData.PaddedHostColSize)
        for (std::size_t i = ii; i < ii + tensorData.GetShape().Cols(); ++i)
        {
            tempPtr[idx] = tensorData.GetDenseHost()[i];
            idx++;
        }

    return tempPtr;
}
} // namespace Sapphire
