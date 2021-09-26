// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire
{
Tensor::Tensor()
    : m_tensorDescKey(-1)
{
}

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

void Tensor::SetMode(DeviceType mode) const
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

    if (desc.Mode() == DeviceType::Cuda)
        desc.ToHost();

    auto dataPtr = std::unique_ptr<float[]>(
        new float[tensorData.GetShape().Size()]);
    std::size_t idx = 0;
    for (std::size_t ii = 0; ii < tensorData.DenseTotalLengthHost;
         ii += tensorData.PaddedHostColSize)
        for (std::size_t i = ii; i < ii + tensorData.GetShape().Cols(); ++i)
        {
            dataPtr[idx] = tensorData.GetDenseHost()[i];
            idx++;
        }

    return dataPtr;
}

std::unique_ptr<float[]> Tensor::GetBackwardDataCopy() const
{
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    const TensorUtil::TensorData tensorData = desc.GetBackwardData();

    if (desc.Mode() == DeviceType::Cuda)
        desc.ToHost();

    auto dataPtr =
        std::unique_ptr<float[]>(new float[tensorData.GetShape().Size()]);
    std::size_t idx = 0;
    for (std::size_t ii = 0; ii < tensorData.DenseTotalLengthHost;
         ii += tensorData.PaddedHostColSize)
        for (std::size_t i = ii; i < ii + tensorData.GetShape().Cols(); ++i)
        {
            dataPtr[idx] = tensorData.GetDenseHost()[i];
            idx++;
        }

    return dataPtr;
}

void Tensor::SetForwardData(std::vector<float> data) const
{
    const auto shape = GetShape();
    if (static_cast<int>(data.size()) != shape.Size())
    {
        throw std::invalid_argument(
            "Tensor::SetForwardData - data size mismatch Given size : (" +
            std::to_string(data.size()) + ") expected size : (" +
            std::to_string(shape.Size()) + ")");
    }
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(
        m_tensorDescKey);

    TensorUtil::TensorData tensorData = desc.GetForwardData();

    std::size_t idx = 0;
    for (std::size_t ii = 0; ii < tensorData.DenseTotalLengthHost;
         ii += tensorData.PaddedHostColSize)
        for (std::size_t i = ii; i < ii + tensorData.GetShape().Cols(); ++i)
        {
            tensorData.GetMutableDenseHost()[i] = data.at(idx);
            idx++;
        }

    if (desc.Mode() == DeviceType::Cuda)
        desc.ToCuda();
}

void Tensor::SetBackwardData(const std::vector<float>& data) const
{
    const auto shape = GetShape();
    if (static_cast<int>(data.size()) != shape.Size())
    {
        throw std::invalid_argument(
            "Tensor::SetForwardData - data size mismatch Given size : (" +
            std::to_string(data.size()) + ") expected size : (" +
            std::to_string(shape.Size()) + ")");
    }
    Model& model = ModelManager::GetCurrentModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);

    TensorUtil::TensorData tensorData = desc.GetBackwardData();

    std::size_t idx = 0;
    for (std::size_t ii = 0; ii < tensorData.DenseTotalLengthHost;
         ii += tensorData.PaddedHostColSize)
        for (std::size_t i = ii; i < ii + tensorData.GetShape().Cols(); ++i)
        {
            tensorData.GetMutableDenseHost()[i] = data.at(idx);
            idx++;
        }

    if (desc.Mode() == DeviceType::Cuda)
        desc.ToCuda();
}
} // namespace Sapphire
