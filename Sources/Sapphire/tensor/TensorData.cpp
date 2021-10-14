// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/compute/dense/cuda/Initialize.cuh>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace Sapphire::TensorUtil
{
TensorData::TensorData(Shape shape, Type type, bool preserve)
    : m_shape(std::move(shape)),
      m_type(type),
      m_preserve(preserve)
{
    m_allocateHost();
}

TensorData::TensorData(Shape shape, Type type, CudaDevice device, bool preserve)
    : m_shape(std::move(shape)),
      m_type(type),
      m_device(std::move(device)),
      m_preserve(preserve)
{
    if (device.GetID() >= 0)
    {
        m_mode = DeviceType::Cuda;
        m_allocateCuda();
    }
    else
        m_allocateHost();
}

TensorData::TensorData(Shape shape, Type type, CudaDevice device,
                       int parentDescKey, bool preserve)
    : m_shape(std::move(shape)),
      m_parentDescKey(parentDescKey),
      m_type(type),
      m_device(std::move(device)),
      m_preserve(preserve)
{
    if (device.GetID() >= 0)
    {
        m_allocateCuda();
        m_mode = DeviceType::Cuda;
    }
    else
        m_allocateHost();
}


TensorData::TensorData(TensorData&& tensorData) noexcept
    : HostTotalSize(tensorData.HostTotalSize),
      DenseTotalLengthCuda(tensorData.DenseTotalLengthCuda),
      SparseTotalLength(tensorData.SparseTotalLength),
      SparseMatHost(tensorData.SparseMatHost),
      SparseMatCuda(tensorData.SparseMatCuda),
      m_shape(std::move(tensorData.m_shape)),
      m_denseHost(tensorData.m_denseHost),
      m_denseCuda(tensorData.m_denseCuda),
      m_parentDescKey(tensorData.m_parentDescKey),
      m_type(tensorData.m_type),
      m_mode(tensorData.m_mode),
      m_device(std::move(tensorData.m_device)),
      m_preserve(tensorData.m_preserve)
{
    tensorData.HostTotalSize = 0;
    tensorData.SparseTotalLength = 0;
    tensorData.m_denseHost = nullptr;
    tensorData.m_denseCuda = nullptr;
    tensorData.SparseMatHost = nullptr;
    tensorData.SparseMatCuda = nullptr;
}

TensorData& TensorData::operator=(TensorData&& tensorData) noexcept
{
    HostTotalSize = tensorData.HostTotalSize;
    SparseTotalLength = tensorData.SparseTotalLength;
    m_denseHost = tensorData.m_denseHost;
    m_denseCuda = tensorData.m_denseCuda;
    SparseMatHost = tensorData.SparseMatHost;
    SparseMatCuda = tensorData.SparseMatCuda;
    m_shape = std::move(tensorData.m_shape);
    m_parentDescKey = tensorData.m_parentDescKey;
    m_type = tensorData.m_type;
    m_mode = tensorData.m_mode;
    m_device = std::move(tensorData.m_device);
    m_preserve = tensorData.m_preserve;

    tensorData.HostTotalSize = 0;
    tensorData.SparseTotalLength = 0;
    tensorData.m_denseHost = nullptr;
    tensorData.m_denseCuda = nullptr;
    tensorData.SparseMatHost = nullptr;
    tensorData.SparseMatCuda = nullptr;

    return *this;
}

void TensorData::Reshape(const Shape& shape)
{
    if (shape.Size() != m_shape.Size())
    {
        throw std::runtime_error(
            "TensorData::Reshape - Attempt to change shape having different "
            "size");
    }

    if (m_mode == DeviceType::Cuda)
    {
        m_shape = shape;
    }

    if (m_mode == DeviceType::Host)
    {
        m_shape = shape;
    }
}

std::vector<float> TensorData::GetDataCopy()
{
    if (m_mode == DeviceType::Cuda)
        m_toHost();

    auto dataPtr = std::vector<float>(m_shape.Size());

    for (std::size_t i = 0; i < HostTotalSize; ++i)
        dataPtr[i] = m_denseHost[i];

    return dataPtr;
}

void TensorData::SetData(std::vector<float> data)
{
    const auto shape = GetShape();
    if (static_cast<int>(data.size()) != shape.Size())
    {
        throw std::invalid_argument(
            "Tensor::SetForwardData - data size mismatch Given size : (" +
            std::to_string(data.size()) + ") expected size : (" +
            std::to_string(shape.Size()) + ")");
    }

    if (m_mode == DeviceType::Cuda)
    {
        Compute::Cuda::CopyHostToDevice(m_denseCuda, &data.front(),
                                        sizeof(float) * shape.Size());
    }

    if (m_mode == DeviceType::Host)
    {
        std::size_t idx = 0;

        for (std::size_t i = 0; i < HostTotalSize; ++i)
            m_denseHost[i] = data.at(idx);
    }
}

int TensorData::GetBatchSize(int requiredDim) const
{
    return m_shape.GetBatchSize(requiredDim);
}

TensorData TensorData::CreateCopy() const
{
    TensorData tensorData(m_shape, GetType(), GetDevice(), m_parentDescKey);
    tensorData.SetMode(m_mode);

    DeepCopy(tensorData, *this);
    return tensorData;
}

void TensorData::SetMode(DeviceType type)
{
    m_mode = type;
    if (m_mode == DeviceType::Host && m_denseHost == nullptr)
        m_allocateHost();
    if (m_mode == DeviceType::Cuda && m_denseCuda == nullptr)
    {
        if (m_device.GetID() == -1)
            throw std::runtime_error(
                "TensorData::SetMode - TensorData has not been configured to  "
                "use cuda device");
        m_allocateCuda();
    }
}

void TensorData::ToCuda()
{
    if (m_denseHost == nullptr && m_mode == DeviceType::Cuda)
        return;
    SetMode(DeviceType::Cuda);
    m_toCuda();
}

void TensorData::ToHost()
{
    if (m_denseCuda == nullptr && m_mode == DeviceType::Host)
        return;
    SetMode(DeviceType::Host);
    m_toHost();
}

void TensorData::DeepCopy(TensorData& dst, const TensorData& src)
{
    if (dst.GetType() != src.GetType())
        throw std::invalid_argument(
            "DeepCopy - matrix or device type mismatch");

    if (dst.Mode() != src.Mode())
        throw std::invalid_argument("DeepCopy - Mode mismatch");

    const auto mode = dst.Mode();
    const auto matrixType = dst.GetType();

    if (mode == DeviceType::Cuda && matrixType == Type::Dense)
    {
        Compute::Cuda::CopyDeviceToDevice(
            dst.m_denseCuda, src.m_denseCuda,
            dst.DenseTotalLengthCuda * sizeof(float));
    }
    else if (mode == DeviceType::Host && matrixType == Type::Dense)
        std::memcpy(dst.m_denseHost, src.m_denseHost,
                    dst.HostTotalSize * sizeof(float));
    else if (mode == DeviceType::Cuda && matrixType == Type::Sparse)
        throw std::runtime_error("DeepCopy - Sparse Not implemented");
    else if (mode == DeviceType::Host && matrixType == Type::Sparse)
        throw std::runtime_error("DeepCopy - Sparse Not implemented");
}

void TensorData::m_toCuda()
{
    if (m_type == Type::Sparse)
    {
        throw std::runtime_error(
            "TensorData::m_toCuda - Sparse matrix not implemented");
    }

    if (m_denseCuda == nullptr)
        m_allocateCuda();

    Compute::Cuda::CopyHostToDevice(m_denseCuda, m_denseHost,
                                    HostTotalSize * sizeof(float));
}

void TensorData::m_toHost()
{
    if (m_type == Type::Sparse)
    {
        throw std::runtime_error(
            "TensorData::m_toHost - Sparse matrix not implemented");
    }

    if (m_denseHost == nullptr)
        m_allocateHost();

    Compute::Cuda::CopyDeviceToHost(m_denseHost, m_denseCuda,
                                    HostTotalSize * sizeof(float));
}

void TensorData::m_allocateHost()
{
    if (m_type == Type::Sparse)
        throw std::runtime_error("m_allocate - Sparse not implemented");

    HostTotalSize = m_shape.Size();

    if (m_preserve)
        m_denseHost = static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(
                HostTotalSize * sizeof(float), true));
    else
        m_denseHost = static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(
                HostTotalSize * sizeof(float)));

    std::memset(m_denseHost, 0, HostTotalSize * sizeof(float));
}

void TensorData::m_allocateCuda()
{
    if (m_type == Type::Sparse)
    {
        throw std::runtime_error("m_allocate - Sparse not implemented");
    }

    const unsigned long totalSize = m_shape.Size();

    if (m_preserve)
    {
        m_denseCuda = static_cast<float*>(
            Util::ResourceManager::GetMemoryCuda(
                totalSize * sizeof(float), true));
    }
    else
        m_denseCuda = static_cast<float*>(
            Util::ResourceManager::GetMemoryCuda(
                totalSize * sizeof(float)));
    DenseTotalLengthCuda = totalSize;

    Compute::Dense::Cuda::Scalar(m_denseCuda, 0.0f, DenseTotalLengthCuda);
}
} // namespace Sapphire::TensorUtil
