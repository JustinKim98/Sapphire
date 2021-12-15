// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire
{
Tensor::Tensor(const Shape& shape, const CudaDevice& device,
               bool preserve)
{
    auto& model = ModelManager::CurModel();
    m_tensorDescKey =
        model.RegisterTensorDescriptor(shape, Type::Dense, device, preserve);
}

Tensor::Tensor(const Shape& shape, const CudaDevice& device,
               Type type, bool preserve)
{
    auto& model = ModelManager::CurModel();
    m_tensorDescKey = model.RegisterTensorDescriptor(
        shape, type, device, preserve);
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
    Model& model = ModelManager::CurModel();
    TensorUtil::TensorDescriptor& desc =
        model.GetDescriptor(m_tensorDescKey);
    return desc.GetShape();
}

CudaDevice Tensor::GetDevice() const
{
    Model& model = ModelManager::CurModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(
        m_tensorDescKey);
    return desc.GetDevice();
}

int Tensor::TensorDescriptorKey() const
{
    return m_tensorDescKey;
}

void Tensor::ToCuda() const
{
    Model& model = ModelManager::CurModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(
        m_tensorDescKey);
    desc.ToCuda();
    desc.SetMode(ComputeMode::Cuda);
}

void Tensor::ToHost() const
{
    Model& model = ModelManager::CurModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    desc.ToHost();
    desc.SetMode(ComputeMode::Host);
}

ComputeMode Tensor::Mode() const
{
    Model& model = ModelManager::CurModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    return desc.Mode();
}

int Tensor::Size() const
{
    return GetShape().Size();
}

void Tensor::SetMode(ComputeMode mode) const
{
    Model& model = ModelManager::CurModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    desc.SetMode(mode);
}

void Tensor::Reshape(Shape shape) const
{
    if (shape.Size() != GetShape().Size())
        throw std::runtime_error(
            "Tensor::Reshape - New shape does not match the size of current "
            "shape");

    Model& model = ModelManager::CurModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);
    desc.Reshape(shape);
}

void Tensor::Flatten() const
{
    const auto newShape = Shape({ GetShape().Size() });
    Reshape(newShape);
}

std::vector<float> Tensor::GetData() const
{
    Model& model = ModelManager::CurModel();
    const TensorUtil::TensorDescriptor& desc =
        model.GetDescriptor(m_tensorDescKey);
    TensorUtil::TensorData tensorData = desc.GetForwardData();

    return tensorData.GetDataCopy();
}

std::vector<float> Tensor::GetGradient() const
{
    Model& model = ModelManager::CurModel();
    const TensorUtil::TensorDescriptor& desc = model.GetDescriptor(
        m_tensorDescKey);
    TensorUtil::TensorData tensorData = desc.GetBackwardData();

    return tensorData.GetDataCopy();
}

void Tensor::LoadData(const std::vector<float>& data) const
{
    const auto shape = GetShape();
    if (static_cast<int>(data.size()) != shape.Size())
    {
        throw std::invalid_argument(
            "Tensor::LoadData - data size mismatch Given size : (" +
            std::to_string(data.size()) + ") expected size : (" +
            std::to_string(shape.Size()) + ")");
    }
    Model& model = ModelManager::CurModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(
        m_tensorDescKey);

    TensorUtil::TensorData tensorData = desc.GetForwardData();
    tensorData.SetData(data);
}

void Tensor::SetGradient(const std::vector<float>& data) const
{
    const auto shape = GetShape();
    if (static_cast<int>(data.size()) != shape.Size())
    {
        throw std::invalid_argument(
            "Tensor::SetGradient - data size mismatch Given size : (" +
            std::to_string(data.size()) + ") expected size : (" +
            std::to_string(shape.Size()) + ")");
    }
    Model& model = ModelManager::CurModel();
    TensorUtil::TensorDescriptor& desc = model.GetDescriptor(m_tensorDescKey);

    TensorUtil::TensorData tensorData = desc.GetBackwardData();

    tensorData.SetData(data);
}
} // namespace Sapphire
