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
    : DenseTotalLengthHost(tensorData.DenseTotalLengthHost),
      DenseTotalLengthCuda(tensorData.DenseTotalLengthCuda),
      SparseTotalLength(tensorData.SparseTotalLength),
      PaddedHostColSize(tensorData.PaddedHostColSize),
      SparseMatHost(tensorData.SparseMatHost),
      SparseMatCuda(tensorData.SparseMatCuda),
      m_shape(std::move(tensorData.m_shape)),
      m_denseMatHost(tensorData.m_denseMatHost),
      m_denseMatCuda(tensorData.m_denseMatCuda),
      m_parentDescKey(tensorData.m_parentDescKey),
      m_type(tensorData.m_type),
      m_mode(tensorData.m_mode),
      m_device(std::move(tensorData.m_device)),
      m_preserve(tensorData.m_preserve)
{
    tensorData.DenseTotalLengthHost = 0;
    tensorData.SparseTotalLength = 0;
    tensorData.m_denseMatHost = nullptr;
    tensorData.m_denseMatCuda = nullptr;
    tensorData.SparseMatHost = nullptr;
    tensorData.SparseMatCuda = nullptr;
}

TensorData& TensorData::operator=(TensorData&& tensorData) noexcept
{
    DenseTotalLengthHost = tensorData.DenseTotalLengthHost;
    SparseTotalLength = tensorData.SparseTotalLength;
    PaddedHostColSize = tensorData.PaddedHostColSize;
    m_denseMatHost = tensorData.m_denseMatHost;
    m_denseMatCuda = tensorData.m_denseMatCuda;
    SparseMatHost = tensorData.SparseMatHost;
    SparseMatCuda = tensorData.SparseMatCuda;
    m_shape = std::move(tensorData.m_shape);
    m_parentDescKey = tensorData.m_parentDescKey;
    m_type = tensorData.m_type;
    m_mode = tensorData.m_mode;
    m_device = std::move(tensorData.m_device);
    m_preserve = tensorData.m_preserve;

    tensorData.DenseTotalLengthHost = 0;
    tensorData.SparseTotalLength = 0;
    tensorData.m_denseMatHost = nullptr;
    tensorData.m_denseMatCuda = nullptr;
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
        const auto data = GetDataCopy();
        m_shape = shape;
        if (m_preserve)
        {
            Util::ResourceManager::FreePreservedHost(m_denseMatHost);
            m_allocateHost();
        }
        SetData(data);
    }
}

std::vector<float> TensorData::GetDataCopy()
{
    if (m_mode == DeviceType::Cuda)
        ToHost();

    auto dataPtr = std::vector<float>(m_shape.Size());
    std::size_t idx = 0;
    for (std::size_t ii = 0; ii < DenseTotalLengthHost; ii += PaddedHostColSize)
        for (std::size_t i = ii; i < ii + m_shape.Cols(); ++i)
        {
            dataPtr[idx] = m_denseMatHost[i];
            idx++;
        }

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
        Compute::Cuda::CopyHostToDevice(m_denseMatCuda, &data.front(),
                                        sizeof(float) * shape.Size());
    }

    if (m_mode == DeviceType::Host)
    {
        std::size_t idx = 0;
        for (std::size_t ii = 0; ii < DenseTotalLengthHost;
             ii += PaddedHostColSize)
            for (std::size_t i = ii; i < ii + m_shape.Cols(); ++i)
            {
                m_denseMatHost[i] = data.at(idx);
                idx++;
            }
    }

    if (m_mode == DeviceType::Cuda)
        ToCuda();
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
    if (m_mode == DeviceType::Host && m_denseMatHost == nullptr)
        m_allocateHost();
    if (m_mode == DeviceType::Cuda && m_denseMatCuda == nullptr)
        m_allocateCuda();
}

void TensorData::ToCuda()
{
    m_toCuda();
}

void TensorData::ToHost()
{
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
            dst.m_denseMatCuda, src.m_denseMatCuda,
            dst.DenseTotalLengthCuda * sizeof(float));
        std::memcpy(dst.m_denseMatHost, src.m_denseMatHost,
                    dst.DenseTotalLengthHost * sizeof(float));
    }
    else if (mode == DeviceType::Host && matrixType == Type::Dense)
        std::memcpy(dst.m_denseMatHost, src.m_denseMatHost,
                    dst.DenseTotalLengthHost * sizeof(float));
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

    if (m_denseMatCuda == nullptr)
        m_allocateCuda();

    Compute::Cuda::CudaSetDevice(m_device.GetID());

    const auto colSize = Cols();
    const auto totalSize = Size();

    for (int rowIdx = 0; rowIdx < totalSize / colSize; rowIdx++)
    {
        auto* cudaPtr = m_denseMatCuda + static_cast<std::size_t>(rowIdx) *
                        colSize;
        auto* hostPtr =
            m_denseMatHost + static_cast<std::size_t>(rowIdx) *
            PaddedHostColSize;

        Compute::Cuda::CopyHostToDevice(cudaPtr, hostPtr,
                                        colSize * sizeof(float));
    }
}

void TensorData::m_toHost()
{
    if (m_type == Type::Sparse)
    {
        throw std::runtime_error(
            "TensorData::m_toHost - Sparse matrix not implemented");
    }

    if (m_denseMatHost == nullptr)
        m_allocateHost();

    const auto colSize = Cols();
    const auto totalSize = m_shape.Size();
    for (int rowIdx = 0; rowIdx < totalSize / colSize; rowIdx++)
    {
        auto* cudaPtr = m_denseMatCuda + rowIdx * colSize;
        auto* hostPtr =
            m_denseMatHost + static_cast<std::size_t>(rowIdx) *
            PaddedHostColSize;
        const size_t bytesToCopy = colSize * sizeof(float);

        Compute::Cuda::CopyDeviceToHost(hostPtr, cudaPtr,
                                        static_cast<unsigned int>(bytesToCopy));
    }
}

void TensorData::m_allocateHost()
{
    const auto colSize = Cols();

    const auto padUnitSize = static_cast<unsigned long>(32 / sizeof(float));

    const auto paddedColumnSize =
        colSize % padUnitSize == 0
            ? colSize
            : colSize / padUnitSize * padUnitSize + padUnitSize;

    if (m_type == Type::Sparse)
    {
        throw std::runtime_error("m_allocate - Sparse not implemented");
    }

    unsigned long totalSize = paddedColumnSize;

    if (m_shape.Dim() > 1)
    {
        for (auto i = 0; i < static_cast<int>(m_shape.Dim()) - 1; ++i)
            totalSize *= m_shape.At(i);
    }

    PaddedHostColSize = paddedColumnSize;
    DenseTotalLengthHost = totalSize;

    if (m_preserve)
    {
        m_denseMatHost = static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(
                totalSize * sizeof(float), true));
    }
    else
        m_denseMatHost = static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(totalSize * sizeof(float)));

    //if (m_mode == DeviceType::Host)
    std::memset(m_denseMatHost, 0, totalSize * sizeof(float));
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
        m_denseMatCuda = static_cast<float*>(
            Util::ResourceManager::GetMemoryCuda(
                totalSize * sizeof(float), true));
    }
    else
        m_denseMatCuda = static_cast<float*>(
            Util::ResourceManager::GetMemoryCuda(
                totalSize * sizeof(float)));
    DenseTotalLengthCuda = totalSize;

    // if (m_mode == DeviceType::Cuda)
    Compute::Dense::Cuda::Scalar(m_denseMatCuda, 0.0f, DenseTotalLengthCuda);
}
} // namespace Sapphire::TensorUtil
