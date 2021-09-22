// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/compute/sparse/Sparse.hpp>
#include <Sapphire/compute/dense/cuda/Initialize.cuh>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace Sapphire::TensorUtil
{
TensorData::TensorData(Shape shape, Type type)
    : TensorShape(std::move(shape)),
      m_type(type)
{
    m_allocateHost();
}

TensorData::TensorData(Shape shape, Type type, CudaDevice device)
    : TensorShape(std::move(shape)),
      m_type(type),
      m_device(std::move(device))
{
    if (device.GetID() >= 0)
        m_allocateCuda();
    m_allocateHost();
}

TensorData::TensorData(Shape shape, Type type, CudaDevice device,
                       int parentDescKey)
    : TensorShape(std::move(shape)),
      m_parentDescKey(parentDescKey),
      m_type(type),
      m_device(std::move(device))
{
    if (device.GetID() >= 0)
        m_allocateCuda();
    m_allocateHost();
}

TensorData::TensorData(const TensorData& tensorData)
    : DenseTotalLengthHost(tensorData.DenseTotalLengthHost),
      DenseTotalLengthCuda(tensorData.DenseTotalLengthCuda),
      SparseTotalLength(tensorData.SparseTotalLength),
      PaddedHostColSize(tensorData.PaddedHostColSize),
      SparseMatHost(tensorData.SparseMatHost),
      SparseMatCuda(tensorData.SparseMatCuda),
      TensorShape(tensorData.TensorShape),
      DenseMatHost(tensorData.DenseMatHost),
      DenseMatCuda(tensorData.DenseMatCuda),
      m_parentDescKey(tensorData.m_parentDescKey),
      m_type(tensorData.m_type),
      m_mode(tensorData.m_mode),
      m_device(tensorData.m_device)
{
    if (DenseMatHost)
    {
        Util::ResourceManager::AddReferenceHost(
            static_cast<void*>(DenseMatHost));
    }
    if (DenseMatCuda)
    {
        Util::ResourceManager::AddReferenceCuda(
            static_cast<void*>(DenseMatCuda));
    }
}

TensorData::TensorData(TensorData&& tensorData) noexcept
    : DenseTotalLengthHost(tensorData.DenseTotalLengthHost),
      DenseTotalLengthCuda(tensorData.DenseTotalLengthCuda),
      SparseTotalLength(tensorData.SparseTotalLength),
      PaddedHostColSize(tensorData.PaddedHostColSize),
      SparseMatHost(tensorData.SparseMatHost),
      SparseMatCuda(tensorData.SparseMatCuda),
      TensorShape(std::move(tensorData.TensorShape)),
      DenseMatHost(tensorData.DenseMatHost),
      DenseMatCuda(tensorData.DenseMatCuda),
      m_parentDescKey(tensorData.m_parentDescKey),
      m_type(tensorData.m_type),
      m_mode(tensorData.m_mode),
      m_device(std::move(tensorData.m_device))
{
    tensorData.DenseTotalLengthHost = 0;
    tensorData.SparseTotalLength = 0;
    tensorData.DenseMatHost = nullptr;
    tensorData.DenseMatCuda = nullptr;
    tensorData.SparseMatHost = nullptr;
    tensorData.SparseMatCuda = nullptr;
}

TensorData& TensorData::operator=(const TensorData& tensorData)
{
    if (this == &tensorData)
        return *this;

    DenseTotalLengthHost = tensorData.DenseTotalLengthHost;
    DenseTotalLengthCuda = tensorData.DenseTotalLengthCuda;
    SparseTotalLength = tensorData.SparseTotalLength;
    PaddedHostColSize = tensorData.PaddedHostColSize;
    DenseMatHost = tensorData.DenseMatHost;
    DenseMatCuda = tensorData.DenseMatCuda;
    SparseMatHost = tensorData.SparseMatHost;
    SparseMatCuda = tensorData.SparseMatCuda;
    TensorShape = tensorData.TensorShape;
    m_parentDescKey = tensorData.m_parentDescKey;
    m_type = tensorData.m_type;
    m_mode = tensorData.m_mode;
    m_device = tensorData.m_device;

    if (DenseMatHost)
    {
        Util::ResourceManager::AddReferenceHost(
            static_cast<void*>(DenseMatHost));
    }
    if (DenseMatCuda)
    {
        Util::ResourceManager::AddReferenceCuda(
            static_cast<void*>(DenseMatCuda));
    }

    return *this;
}

TensorData& TensorData::operator=(TensorData&& tensorData) noexcept
{
    DenseTotalLengthHost = tensorData.DenseTotalLengthHost;
    SparseTotalLength = tensorData.SparseTotalLength;
    PaddedHostColSize = tensorData.PaddedHostColSize;
    DenseMatHost = tensorData.DenseMatHost;
    DenseMatCuda = tensorData.DenseMatCuda;
    SparseMatHost = tensorData.SparseMatHost;
    SparseMatCuda = tensorData.SparseMatCuda;
    TensorShape = std::move(tensorData.TensorShape);
    m_parentDescKey = tensorData.m_parentDescKey;
    m_type = tensorData.m_type;
    m_mode = tensorData.m_mode;
    m_device = std::move(tensorData.m_device);

    tensorData.DenseTotalLengthHost = 0;
    tensorData.SparseTotalLength = 0;
    tensorData.DenseMatHost = nullptr;
    tensorData.DenseMatCuda = nullptr;
    tensorData.SparseMatHost = nullptr;
    tensorData.SparseMatCuda = nullptr;

    return *this;
}

TensorData::~TensorData()
{
    m_freeHost();
    m_freeCuda();
}

unsigned TensorData::GetBatchSize(unsigned int requiredDim) const
{
    return TensorShape.GetBatchSize(requiredDim);
}

TensorData TensorData::CreateCopy() const
{
    TensorData tensorData(TensorShape, GetType(), GetDevice(),
                          m_parentDescKey);
    tensorData.SetMode(m_mode);

    DeepCopy(tensorData, *this);
    return tensorData;
}

void TensorData::SetMode(DeviceType type)
{
    m_mode = type;
}

void TensorData::ToCuda() const
{
    m_toCuda(*this);
}

void TensorData::ToHost() const
{
    m_toHost(*this);
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
            dst.DenseMatCuda, src.DenseMatCuda,
            dst.DenseTotalLengthCuda * sizeof(float));
        std::memcpy(dst.DenseMatHost, src.DenseMatHost,
                    dst.DenseTotalLengthHost * sizeof(float));
    }
    else if (mode == DeviceType::Host && matrixType == Type::Dense)
        std::memcpy(dst.DenseMatHost, src.DenseMatHost,
                    dst.DenseTotalLengthHost * sizeof(float));
        //! TODO : implement Sparse copy and allocation for TensorData
    else if (mode == DeviceType::Cuda && matrixType == Type::Sparse)
        throw std::runtime_error("DeepCopy - Sparse Not implemented");
    else if (mode == DeviceType::Host && matrixType == Type::Sparse)
        throw std::runtime_error("DeepCopy - Not implemented");
}

void TensorData::m_toCuda(const TensorData& tensorData)
{
    if (tensorData.GetType() == Type::Sparse)
    {
        throw std::runtime_error(
            "TensorData::m_toCuda - Sparse matrix not implemented");
    }

    Compute::Cuda::CudaSetDevice(tensorData.m_device.GetID());

    const auto colSize = tensorData.Cols();
    const auto totalSize = tensorData.TensorShape.Size();

    for (size_t rowIdx = 0; rowIdx < totalSize / colSize; rowIdx++)
    {
        float* cudaPtr = tensorData.DenseMatCuda + rowIdx * colSize;
        float* hostPtr =
            tensorData.DenseMatHost + rowIdx * tensorData.PaddedHostColSize;
        const unsigned int bytesToCopy = colSize * sizeof(float);

        Compute::Cuda::CopyHostToDevice(cudaPtr, hostPtr,
                                        bytesToCopy);
    }
}

void TensorData::m_toHost(const TensorData& tensorData)
{
    if (tensorData.GetType() == Type::Sparse)
    {
        throw std::runtime_error(
            "TensorData::m_toHost - Sparse matrix not implemented");
    }

    const auto colSize = tensorData.Cols();
    const auto totalSize = tensorData.TensorShape.Size();
    for (size_t rowIdx = 0; rowIdx < totalSize / colSize; rowIdx++)
    {
        float* cudaPtr = tensorData.DenseMatCuda + rowIdx * colSize;
        float* hostPtr =
            tensorData.DenseMatHost + rowIdx * tensorData.PaddedHostColSize;
        const size_t bytesToCopy = colSize * sizeof(float);

        Compute::Cuda::CopyDeviceToHost(hostPtr, cudaPtr,
                                        static_cast<unsigned int>(bytesToCopy));
    }
}

void TensorData::m_freeHost()
{
    if (m_type == Type::Sparse && SparseMatHost)
    {
        delete[] SparseMatHost;
    }
    else if (DenseMatHost)
    {
        Util::ResourceManager::DeReferenceHost(
            static_cast<void*>(DenseMatHost));
        DenseTotalLengthHost = 0;
    }
}

void TensorData::m_freeCuda()
{
    if (m_device.GetID() >= 0)
        Compute::Cuda::CudaSetDevice(m_device.GetID());

    if (m_type == Type::Sparse && SparseMatCuda)
    {
        Compute::Cuda::CudaFree(static_cast<void*>(SparseMatCuda));
    }
    else if (DenseMatCuda)
    {
        Util::ResourceManager::DeReferenceCuda(static_cast<void*>(DenseMatCuda),
                                               m_device.GetID());
        DenseTotalLengthCuda = 0;
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

    if (TensorShape.Dim() > 1)
    {
        for (auto i = 0; i < static_cast<int>(TensorShape.Dim()) - 1; ++i)
            totalSize *= TensorShape.At(i);
    }

    PaddedHostColSize = paddedColumnSize;
    DenseTotalLengthHost = totalSize;

    DenseMatHost =
        static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(
                totalSize * sizeof(float)));

    std::memset(DenseMatHost, 0, totalSize * sizeof(float));
}

void TensorData::m_allocateCuda()
{
    if (m_type == Type::Sparse)
    {
        throw std::runtime_error("m_allocate - Sparse not implemented");
    }

    cudaSetDevice(m_device.GetID());
    const unsigned long totalSize = TensorShape.Size();
    DenseMatCuda = static_cast<float*>(Util::ResourceManager::GetMemoryCuda(
        totalSize * sizeof(float), m_device.GetID()));
    DenseTotalLengthCuda = totalSize;

    Compute::Dense::Cuda::Scalar(DenseMatCuda, 0.0f, DenseTotalLengthCuda);
}
} // namespace Sapphire::TensorUtil
