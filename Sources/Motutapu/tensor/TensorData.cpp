// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <immintrin.h>
#include <Motutapu/compute/cuda/Memory.hpp>
#include <Motutapu/tensor/TensorData.hpp>
#include <Motutapu/util/MemoryManager.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace Motutapu::TensorUtil
{
TensorData::TensorData(Shape shape, Type type, Device device,
                       unsigned int batchSize)
    : BatchSize(batchSize),
      TensorShape(std::move(shape)),
      m_type(type),
      m_device(std::move(device))
{
    if (device.Type() == DeviceType::CUDA)
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
    if (device.Type() == DeviceType::CUDA)
    {
        m_allocateCuda(batchSize);
    }
    m_allocateHost(batchSize);
}

TensorData::TensorData(const TensorData &tensorData)
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
        Util::MemoryManager::AddReferenceHost(
            static_cast<void *>(DenseMatHost));
    }
    if (DenseMatCuda)
    {
        Util::MemoryManager::AddReferenceCuda(static_cast<void *>(DenseMatCuda),
                                              m_device.GetID());
    }
}

TensorData::TensorData(TensorData &&tensorData) noexcept
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

TensorData &TensorData::operator=(const TensorData &tensorData)
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
        Util::MemoryManager::AddReferenceHost(
            static_cast<void *>(DenseMatHost));
    }
    if (DenseMatCuda)
    {
        Util::MemoryManager::AddReferenceCuda(static_cast<void *>(DenseMatCuda),
                                              m_device.GetID());
    }

    return *this;
}

TensorData &TensorData::operator=(TensorData &&tensorData) noexcept
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

bool TensorData::CopyTensorData(TensorData dest, const TensorData &src)
{
    if (src.GetDevice() != dest.GetDevice())
    {
        throw std::invalid_argument("Device mismatch while copying tensorData");
    }

    if (dest.TensorShape != src.TensorShape)
    {
        throw std::invalid_argument("Shape mismatch while copying tensorData");
    }

    if (dest.GetType() != src.GetType())
    {
        throw std::invalid_argument("Type mismatch while copying tensorData");
    }

    const bool sparse = src.GetType() == Type::Sparse;
    bool success = true;
    const auto device = src.GetDevice();

    const Shape shape = dest.TensorShape;

    if (device.Type() == DeviceType::HOST)
    {
        if (sparse)
        {
            throw std::runtime_error("CopyTensorData - sparse not implemented");
        }
        else
        {
            std::memcpy(dest.DenseMatHost, src.DenseMatHost,
                        src.DenseTotalLengthHost * sizeof(float));
            dest.DenseTotalLengthHost = src.DenseTotalLengthHost;
        }
    }

    if (device.Type() == DeviceType::CUDA)
    {
        success &= Compute::Cuda::CudaSetDevice(device.GetID());

        if (sparse)
        {
            throw std::runtime_error("CopyTensorData - sparse not implemented");
        }
        else
        {
            Compute::Cuda::CopyDeviceToDevice(
                dest.DenseMatCuda, src.DenseMatCuda,
                src.DenseTotalLengthCuda * sizeof(float));
            dest.DenseTotalLengthCuda = src.DenseTotalLengthCuda;
        }
    }

    return success;
}

TensorData TensorData::CreateCopy() const
{
    TensorData tensorData(TensorShape, GetType(), GetDevice(), BatchSize,
                          m_parentDescKey);

    DeepCopy(tensorData, *this);
    return tensorData;
}

bool TensorData::SendTo(const Device &device)
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
        TensorData::m_toGpu(*this);
    }
    else if (m_device.Type() == DeviceType::CUDA &&
             device.Type() == DeviceType::HOST)
    {
        TensorData::m_toHost(*this);
        m_freeCuda();
        m_device = device;
    }
    else if (m_device.Type() == DeviceType::CUDA &&
             device.Type() == DeviceType::CUDA)
    {
        TensorData::m_toHost(*this);
        m_device = device;
        TensorData::m_toGpu(*this);
    }

    return true;
}

void TensorData::DeepCopy(TensorData &dst, const TensorData &src)
{
    if (dst.GetType() != src.GetType())
        throw std::invalid_argument(
            "DeepCopy - matrix or device type mismatch");

    if (dst.GetDevice().Type() != src.GetDevice().Type())
        throw std::invalid_argument("DeepCopy - Device type mismatch");

    auto deviceType = dst.GetDevice().Type();
    auto matrixType = dst.GetType();

    if (deviceType == DeviceType::CUDA && matrixType == Type::Dense)
        Compute::Cuda::CopyDeviceToDevice(
            dst.DenseMatCuda, src.DenseMatCuda,
            dst.DenseTotalLengthCuda * sizeof(float));

    else if (deviceType == DeviceType::CUDA && matrixType == Type::Sparse)
        throw std::runtime_error("DeepCopy - Not implemented");

    else if (deviceType == DeviceType::HOST && matrixType == Type::Dense)
        std::memcpy(dst.DenseMatHost, src.DenseMatHost,
                    dst.DenseTotalLengthHost);

    else if (deviceType == DeviceType::HOST && matrixType == Type::Sparse)
        throw std::runtime_error("DeepCopy - Not implemented");
}

void TensorData::m_toGpu(const TensorData &tensorData)
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
    else
    {
        cudaDeviceSynchronize();

        if (!Compute::Cuda::CudaSetDevice(tensorData.m_device.GetID()))
        {
            throw std::runtime_error("m_toHost - illegalDeviceID");
        }

        const auto colSize = tensorData.Cols();
        const auto totalSize =
            tensorData.TensorShape.Size() * tensorData.BatchSize;

        for (size_t rowIdx = 0; rowIdx < totalSize / colSize; rowIdx++)
        {
            float *cudaPtr = tensorData.DenseMatCuda + rowIdx * colSize;
            float *hostPtr =
                tensorData.DenseMatHost + rowIdx * tensorData.PaddedHostColSize;
            const size_t bytesToCopy = colSize * sizeof(float);

            if (!Compute::Cuda::CopyHostToDevice((void *)cudaPtr,
                                                 (void *)hostPtr, bytesToCopy))
            {
                throw std::runtime_error("m_toGpu - cudaMemCopy failed");
            }
        }
        cudaDeviceSynchronize();
    }
}

void TensorData::m_toHost(const TensorData &tensorData)
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
    else
    {
        cudaDeviceSynchronize();

        const auto colSize = tensorData.Cols();
        const auto totalSize =
            tensorData.TensorShape.Size() * tensorData.BatchSize;

        for (size_t rowIdx = 0; rowIdx < totalSize / colSize; rowIdx++)
        {
            float *cudaPtr = tensorData.DenseMatCuda + rowIdx * colSize;
            float *hostPtr =
                tensorData.DenseMatHost + rowIdx * tensorData.PaddedHostColSize;
            const size_t bytesToCopy = colSize * sizeof(float);

            if (!Compute::Cuda::CopyDeviceToHost((void *)hostPtr,
                                                 (void *)cudaPtr, bytesToCopy))
            {
                throw std::runtime_error("m_toGpu - cudaMemCopy failed");
            }
        }
        cudaDeviceSynchronize();
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
        Util::MemoryManager::DeReferenceHost(static_cast<void *>(DenseMatHost));
        DenseTotalLengthHost = 0;
    }
}

bool TensorData::m_freeCuda()
{
    bool isSuccess = true;

    isSuccess &= Compute::Cuda::CudaSetDevice(m_device.GetID());

    if (m_type == Type::Sparse && SparseMatCuda)
    {
        isSuccess &= Compute::Cuda::CudaFree((void *)(SparseMatCuda));
    }
    else if (DenseMatCuda)
    {
        //         Compute::Cuda::CudaFree((void *)DenseMatCuda);
        Util::MemoryManager::DeReferenceCuda(static_cast<void *>(DenseMatCuda),
                                             m_device.GetID());
        DenseTotalLengthCuda = 0;
    }

    return isSuccess;
}

void TensorData::m_allocateHost(unsigned int batchSize)
{
    const auto colSize = Cols();

    const auto padUnitSize = 32 / sizeof(float);

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
        size_t totalSize = paddedColumnSize;

        if (TensorShape.Dim() > 1)
        {
            for (auto i = 0; i < static_cast<int>(TensorShape.Dim()) - 1; ++i)
                totalSize *= TensorShape.At(i);
        }
        totalSize *= batchSize;

        PaddedHostColSize = paddedColumnSize;
        DenseTotalLengthHost = totalSize;
        DenseMatHost = static_cast<float *>(
            Util::MemoryManager::GetMemoryHost(totalSize * sizeof(float)));

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < totalSize / padUnitSize; ++i)
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
    else
    {
        size_t totalSize = TensorShape.Size() * batchSize;
        DenseMatCuda = static_cast<float *>(Util::MemoryManager::GetMemoryCuda(
            totalSize * sizeof(float), m_device.GetID()));
    }
}

}  // namespace Motutapu::TensorUtil
