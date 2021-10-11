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
      PaddedHostColSize(tensorData.PaddedHostColSize),
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
    PaddedHostColSize = tensorData.PaddedHostColSize;
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
        const auto data = GetDataCopy();
        m_shape = shape;

        const auto colSize = Cols();
        constexpr auto padUnitSize =
            static_cast<unsigned long>(32 / sizeof(float));
        const auto paddedColumnSize =
            colSize % padUnitSize == 0
                ? colSize
                : colSize / padUnitSize * padUnitSize + padUnitSize;
        unsigned long totalSize = paddedColumnSize;

        if (m_shape.Dim() > 1)
        {
            for (auto i = 0; i < static_cast<int>(m_shape.Dim()) - 1; ++i)
                totalSize *= m_shape.At(i);
        }
        PaddedHostColSize = paddedColumnSize;
        HostTotalSize = totalSize;

        if (m_preserve)
        {
            Util::ResourceManager::FreePreservedHost(m_denseHost);
            m_allocateHost();
        }
        SetData(data);
    }
}

std::vector<float> TensorData::GetDataCopy()
{
    if (m_mode == DeviceType::Cuda)
        m_toHost();

    auto dataPtr = std::vector<float>(m_shape.Size());
    std::size_t idx = 0;
    for (std::size_t ii = 0; ii < HostTotalSize; ii += PaddedHostColSize)
        for (std::size_t i = ii; i < ii + m_shape.Cols(); ++i)
        {
            dataPtr[idx] = m_denseHost[i];
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
        Compute::Cuda::CopyHostToDevice(m_denseCuda, &data.front(),
                                        sizeof(float) * shape.Size());
    }

    if (m_mode == DeviceType::Host)
    {
        std::size_t idx = 0;
        for (std::size_t ii = 0; ii < HostTotalSize;
             ii += PaddedHostColSize)
            for (std::size_t i = ii; i < ii + m_shape.Cols(); ++i)
            {
                m_denseHost[i] = data.at(idx);
                idx++;
            }
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

    Compute::Cuda::CudaSetDevice(m_device.GetID());

    const auto colSize = Cols();
    const auto totalSize = Size();

    for (int rowIdx = 0; rowIdx < totalSize / colSize; rowIdx++)
    {
        auto* cudaPtr = m_denseCuda + static_cast<std::size_t>(rowIdx) *
                        colSize;
        auto* hostPtr =
            m_denseHost + static_cast<std::size_t>(rowIdx) *
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

    if (m_denseHost == nullptr)
        m_allocateHost();

    const auto colSize = Cols();
    const auto totalSize = m_shape.Size();
    for (int rowIdx = 0; rowIdx < totalSize / colSize; rowIdx++)
    {
        auto* cudaPtr = m_denseCuda + rowIdx * colSize;
        auto* hostPtr =
            m_denseHost + static_cast<std::size_t>(rowIdx) *
            PaddedHostColSize;
        const size_t bytesToCopy = colSize * sizeof(float);

        Compute::Cuda::CopyDeviceToHost(hostPtr, cudaPtr,
                                        static_cast<unsigned int>(bytesToCopy));
    }
}

void TensorData::m_allocateHost()
{
    if (m_type == Type::Sparse)
    {
        throw std::runtime_error("m_allocate - Sparse not implemented");
    }
    const auto colSize = Cols();
    constexpr auto padUnitSize = static_cast<unsigned long>(32 / sizeof(float));
    const auto paddedColumnSize =
        colSize % padUnitSize == 0
            ? colSize
            : colSize / padUnitSize * padUnitSize + padUnitSize;
    unsigned long totalSize = paddedColumnSize;

    if (m_shape.Dim() > 1)
    {
        for (auto i = 0; i < static_cast<int>(m_shape.Dim()) - 1; ++i)
            totalSize *= m_shape.At(i);
    }

    PaddedHostColSize = paddedColumnSize;
    HostTotalSize = totalSize;

    if (m_preserve)
    {
        m_denseHost = static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(
                totalSize * sizeof(float), true));
    }
    else
        m_denseHost = static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(totalSize * sizeof(float)));

    //if (m_mode == DeviceType::Host)
    std::memset(m_denseHost, 0, totalSize * sizeof(float));
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

    //if (m_mode == DeviceType::Cuda)
    Compute::Dense::Cuda::Scalar(m_denseCuda, 0.0f, DenseTotalLengthCuda);
}
} // namespace Sapphire::TensorUtil
