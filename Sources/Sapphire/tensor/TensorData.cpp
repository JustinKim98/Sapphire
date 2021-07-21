// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <immintrin.h>
#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/compute/sparse/Sparse.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace Sapphire::TensorUtil
{
TensorData::TensorData(Shape shape, Type type, Device device,
                       unsigned int batchSize)
    : BatchSize(batchSize),
      TensorShape(std::move(shape)),
      m_type(type),
      m_device(std::move(device))
{
    if (m_device.Type() == DeviceType::CUDA)
    {
        m_allocateCuda(batchSize);
    }
    m_allocateHost(batchSize);
}

TensorData::TensorData(Shape shape, Type type, Device device,
                       unsigned int batchSize, int parentDescKey)
    : BatchSize(batchSize),
      TensorShape(std::move(shape)),
      m_parentDescKey(parentDescKey),
      m_type(type),
      m_device(std::move(device))
{
    if (m_device.Type() == DeviceType::CUDA)
    {
        m_allocateCuda(batchSize);
    }
    m_allocateHost(batchSize);
}

TensorData::TensorData(const TensorData& tensorData)
    : DenseTotalLengthHost(tensorData.DenseTotalLengthHost),
      DenseTotalLengthCuda(tensorData.DenseTotalLengthCuda),
      SparseTotalLength(tensorData.SparseTotalLength),
      PaddedHostColSize(tensorData.PaddedHostColSize),
      BatchSize(tensorData.BatchSize),
      DenseMatHost(tensorData.DenseMatHost),
      DenseMatCuda(tensorData.DenseMatCuda),
      SparseMatHost(tensorData.SparseMatHost),
      SparseMatCuda(tensorData.SparseMatCuda),
      TensorShape(tensorData.TensorShape),
      m_type(tensorData.m_type),
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
      BatchSize(tensorData.BatchSize),
      DenseMatHost(tensorData.DenseMatHost),
      DenseMatCuda(tensorData.DenseMatCuda),
      SparseMatHost(tensorData.SparseMatHost),
      SparseMatCuda(tensorData.SparseMatCuda),
      TensorShape(std::move(tensorData.TensorShape)),
      m_type(tensorData.m_type),
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
    BatchSize = tensorData.BatchSize;
    DenseMatHost = tensorData.DenseMatHost;
    DenseMatCuda = tensorData.DenseMatCuda;
    SparseMatHost = tensorData.SparseMatHost;
    SparseMatCuda = tensorData.SparseMatCuda;
    TensorShape = tensorData.TensorShape;
    m_type = tensorData.m_type;
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
    BatchSize = tensorData.BatchSize;
    DenseMatHost = tensorData.DenseMatHost;
    DenseMatCuda = tensorData.DenseMatCuda;
    SparseMatHost = tensorData.SparseMatHost;
    SparseMatCuda = tensorData.SparseMatCuda;
    TensorShape = std::move(tensorData.TensorShape);
    m_type = tensorData.m_type;
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

    if (m_device.Type() == DeviceType::CUDA)
    {
        m_freeCuda();
    }
}

TensorData TensorData::CreateCopy() const
{
    TensorData tensorData(TensorShape, GetType(), GetDevice(), BatchSize,
                          m_parentDescKey);

    DeepCopy(tensorData, *this);
    return tensorData;
}

bool TensorData::SendTo(const Device& device)
{
    if (m_device == device)
    {
        return false;
    }

    if (m_device.Type() == DeviceType::HOST &&
        device.Type() == DeviceType::CUDA)
    {
        m_device = device;
        m_allocateCuda(BatchSize);
        m_toGpu(*this);
    }
    else if (m_device.Type() == DeviceType::CUDA &&
             device.Type() == DeviceType::HOST)
    {
        m_toHost(*this);
        m_freeCuda();
        m_device = device;
    }
    else if (m_device.Type() == DeviceType::CUDA &&
             device.Type() == DeviceType::CUDA)
    {
        m_toHost(*this);
        m_device = device;
        m_toGpu(*this);
    }

    return true;
}

void TensorData::DeepCopy(TensorData& dst, const TensorData& src)
{
    if (dst.GetType() != src.GetType())
        throw std::invalid_argument(
            "DeepCopy - matrix or device type mismatch");

    if (dst.GetDevice().Type() != src.GetDevice().Type())
        throw std::invalid_argument("DeepCopy - Device type mismatch");

    const auto deviceType = dst.GetDevice().Type();
    const auto matrixType = dst.GetType();

    if (deviceType == DeviceType::CUDA && matrixType == Type::Dense)
        Compute::Cuda::CopyDeviceToDevice(
            dst.DenseMatCuda, src.DenseMatCuda,
            dst.DenseTotalLengthCuda * sizeof(float));

        //! TODO : implement Sparse copy and allocation for TensorData
    else if (deviceType == DeviceType::CUDA && matrixType == Type::Sparse)
        throw std::runtime_error("DeepCopy - Sparse Not implemented");

    else if (deviceType == DeviceType::HOST && matrixType == Type::Dense)
        std::memcpy(dst.DenseMatHost, src.DenseMatHost,
                    dst.DenseTotalLengthHost);

    else if (deviceType == DeviceType::HOST && matrixType == Type::Sparse)
        throw std::runtime_error("DeepCopy - Not implemented");
}

void TensorData::m_toGpu(const TensorData& tensorData)
{
    if (tensorData.GetDevice().Type() != DeviceType::CUDA)
    {
        throw std::invalid_argument(
            "m_toGpu - Given tensor data is not GPU tensor");
    }

    if (tensorData.GetType() == Type::Sparse)
    {
        throw std::runtime_error("Sparse matrix not implemented");
    }

    Compute::Cuda::CudaSetDevice(tensorData.m_device.GetID());

    const auto colSize = tensorData.Cols();
    const auto totalSize =
        tensorData.TensorShape.Size() * tensorData.BatchSize;

    for (size_t rowIdx = 0; rowIdx < totalSize / colSize; rowIdx++)
    {
        float* cudaPtr = tensorData.DenseMatCuda + rowIdx * colSize;
        float* hostPtr =
            tensorData.DenseMatHost + rowIdx * tensorData.PaddedHostColSize;
        const size_t bytesToCopy = colSize * sizeof(float);

        Compute::Cuda::CopyHostToDevice((void*)cudaPtr, (void*)hostPtr,
                                        bytesToCopy);
    }
}

void TensorData::m_toHost(const TensorData& tensorData)
{
    if (tensorData.GetDevice().Type() != DeviceType::CUDA)
    {
        throw std::invalid_argument(
            "m_toGpu - Given tensor data is not GPU tensor");
    }

    if (tensorData.GetType() == Type::Sparse)
    {
        throw std::runtime_error("Sparse matrix not implemented");
    }

    const auto colSize = tensorData.Cols();
    const auto totalSize =
        tensorData.TensorShape.Size() * tensorData.BatchSize;

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
    Compute::Cuda::CudaSetDevice(m_device.GetID());

    if (m_type == Type::Sparse && SparseMatCuda)
    {
        Compute::Cuda::CudaFree((void*)(SparseMatCuda));
    }
    else if (DenseMatCuda)
    {
        //         Compute::Cuda::CudaFree((void *)DenseMatCuda);
        Util::ResourceManager::DeReferenceCuda(static_cast<void*>(DenseMatCuda),
                                               m_device.GetID());
        DenseTotalLengthCuda = 0;
    }
}

void TensorData::m_allocateHost(unsigned int batchSize)
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
    else
    {
        unsigned long totalSize = paddedColumnSize;

        if (TensorShape.Dim() > 1)
        {
            for (auto i = 0; i < static_cast<int>(TensorShape.Dim()) - 1; ++i)
                totalSize *= TensorShape.At(i);
        }
        totalSize *= batchSize;

        PaddedHostColSize = paddedColumnSize;
        DenseTotalLengthHost = totalSize;
        DenseMatHost = static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(totalSize * sizeof(float)));

#pragma omp parallel for default(none) schedule(static) \
    shared(totalSize, padUnitSize)
        for (long i = 0; i < static_cast<long>(totalSize / padUnitSize); ++i)
            _mm256_store_ps(DenseMatHost + i * padUnitSize,
                            _mm256_set1_ps(0.0f));
    }
}

void TensorData::m_allocateCuda(unsigned int batchSize)
{
    if (m_device.Type() != DeviceType::CUDA)
    {
        throw std::runtime_error(
            "m_allocateCuda - Tensor Device type is not CUDA");
    }

    if (m_type == Type::Sparse)
    {
        throw std::runtime_error("m_allocate - Sparse not implemented");
    }

    cudaSetDevice(m_device.GetID());
    const unsigned long totalSize = TensorShape.Size() * batchSize;
    DenseMatCuda = static_cast<float*>(Util::ResourceManager::GetMemoryCuda(
        totalSize * sizeof(float), m_device.GetID()));
}
} // namespace Sapphire::TensorUtil
