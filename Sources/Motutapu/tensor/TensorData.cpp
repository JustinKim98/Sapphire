// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/Memory.cuh>
#include <Motutapu/tensor/TensorData.hpp>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace Motutapu::Util
{
TensorData::TensorData(Shape shape, Type type, Device device,
                       unsigned int batchSize)
    : BatchSize(batchSize),
      TensorShape(std::move(shape)),
      m_type(type),
      m_device(std::move(device))
{
    auto success = true;

    //! todo : Change this to use memory manager

    if (device.Type() == DeviceType::CUDA)
    {
        success &= Compute::Cuda::CudaSetDevice(device.GetID());
        success &= m_allocateCuda(batchSize);
        m_allocateCpu(batchSize);
    }

    if (!success)
    {
        throw std::runtime_error("Tensor creation failed");
    }
}

TensorData::~TensorData()
{
    m_freeCpu();

    if (m_device.Type() == DeviceType::CUDA)
    {
        m_freeGpu();
    }
}

void TensorData::DenseToSparse(TensorData tensorData)
{
    throw std::runtime_error("DenseToSparse not implemented");
}

void TensorData::SparseToDense(TensorData tensorData)
{
    throw std::runtime_error("SparseToDense not implemented");
}

bool TensorData::CopyTensorData(TensorData dest, const TensorData src)
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

    const bool sparse = src.GetType() == Type::Sparse ? true : false;
    bool success = true;
    const auto device = src.GetDevice();

    const Shape shape = dest.TensorShape;

    if (device.Type() == DeviceType::CPU)
    {
        if (sparse)
        {
            throw std::runtime_error("CopyTensorData - sparse not implemented");
        }
        else
        {
            std::memcpy(dest.DenseMatHost, src.DenseMatHost,
                        src.DenseTotalLength * sizeof(float));
            dest.DenseTotalLength = src.DenseTotalLength;
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
            Compute::Cuda::MemcpyGpuToGpu(dest.DenseMatCuda, src.DenseMatCuda,
                                          src.DenseTotalLength);
            dest.DenseTotalLength = src.DenseTotalLength;
        }
    }

    return success;
}

bool TensorData::ChangeDevice(TensorData tensorData, Device device)
{
    const auto currentDevice = tensorData.GetDevice();
    if (currentDevice == device)
    {
        return false;
    }

    if (currentDevice.Type() == DeviceType::CPU &&
        device.Type() == DeviceType::CUDA)
    {
        tensorData.m_allocateCuda(tensorData.BatchSize);
        TensorData::CopyHostToGpu(tensorData);
    }

    if (currentDevice.Type() == DeviceType::CUDA &&
        device.Type() == DeviceType::CPU)
    {
        TensorData::CopyGpuToHost(tensorData);
    }

    if (currentDevice.Type() == DeviceType::CUDA &&
        device.Type() == DeviceType::CUDA)
    {
        TensorData::CopyGpuToHost(tensorData);
        TensorData::CopyHostToGpu(tensorData);
    }

    tensorData.m_device = device;

    return true;
}

void TensorData::CopyHostToGpu(TensorData tensorData)
{
    if (tensorData.GetDevice().Type() != DeviceType::CUDA)
    {
        throw std::invalid_argument(
            "CopyHostToGpu - Given tensor data is not GPU tensor");
    }

    if (tensorData.GetType() == Type::Sparse)
    {
        throw std::runtime_error("Sparse matrix not implemented");
    }
    else
    {
        Compute::Cuda::CudaSetDevice(tensorData.m_device.GetID());

        Compute::Cuda::MemcpyHostToGpu(tensorData.DenseMatCuda,
                                       tensorData.DenseMatHost,
                                       tensorData.BatchSize);
    }
}

void TensorData::CopyGpuToHost(TensorData tensorData)
{
    if (tensorData.GetDevice().Type() != DeviceType::CUDA)
    {
        throw std::invalid_argument(
            "CopyHostToGpu - Given tensor data is not GPU tensor");
    }

    if (tensorData.GetType() == Type::Sparse)
    {
        throw std::runtime_error("Sparse matrix not implemented");
    }
    else
    {
        Compute::Cuda::CudaSetDevice(tensorData.m_device.GetID());

        Compute::Cuda::MemcpyGpuToHost(tensorData.DenseMatHost,
                                       tensorData.DenseMatCuda,
                                       tensorData.BatchSize);
    }
}

void TensorData::m_freeCpu() const
{
    if (m_type == Type::Sparse)
    {
        delete[] SparseMatHost;
    }
    else
    {
        delete[] DenseMatHost;
    }
}

bool TensorData::m_freeGpu()
{
    bool isSuccess = true;

    isSuccess &= Compute::Cuda::CudaSetDevice(m_device.GetID());

    if (m_type == Type::Sparse)
    {
        isSuccess &=
            Compute::Cuda::CudaFree(reinterpret_cast<void**>(SparseMatCuda));
    }
    else
    {
        isSuccess &=
            Compute::Cuda::CudaFree(reinterpret_cast<void**>(DenseMatCuda));
    }

    return isSuccess;
}

void TensorData::m_allocateCpu(unsigned int batchSize)
{
    const auto colSize = TensorShape.At(0);
    const auto rowSize = TensorShape.Dim() > 1 ? TensorShape.At(1) : 0;

    const auto padUnitSize = static_cast<unsigned int>(32 / sizeof(float));

    const auto paddedColSize =
        colSize % padUnitSize == 0
            ? colSize
            : colSize / padUnitSize * padUnitSize + padUnitSize;

    const auto paddedRowSize =
        rowSize % padUnitSize == 0
            ? rowSize
            : rowSize / padUnitSize * padUnitSize + padUnitSize;

    if (m_type == Type::Sparse)
    {
        throw std::runtime_error("m_allocate - Sparse not implemented");
    }
    else
    {
        DenseTotalLength = batchSize * paddedRowSize * paddedColSize;
        DenseMatHost = new float[DenseTotalLength];
    }
}

bool TensorData::m_allocateCuda(unsigned int batchSize)
{
    const auto colSize = TensorShape.At(0);
    const auto rowSize = TensorShape.Dim() > 1 ? TensorShape.At(1) : 0;

    const unsigned int padUnitSize = 32 / sizeof(float);

    const auto paddedColSize =
        colSize % padUnitSize == 0
            ? colSize
            : colSize / padUnitSize * padUnitSize + padUnitSize;

    const auto paddedRowSize =
        rowSize % padUnitSize == 0
            ? rowSize
            : rowSize / padUnitSize * padUnitSize + padUnitSize;

    auto isSuccess = true;

    if (m_device.Type() == DeviceType::CUDA)
    {
        isSuccess &= Compute::Cuda::CudaSetDevice(m_device.GetID());
        if (m_type == Type::Sparse)
        {
            throw std::runtime_error("m_allocate - Sparse not implemented");
        }
        else
        {
            isSuccess &= Compute::Cuda::CudaMalloc(
                &DenseMatCuda, batchSize * paddedRowSize * paddedColSize);
        }
    }
    else
    {
        throw std::runtime_error(
            "m_allocateCuda - Tensor Data type is not CUDA");
    }

    return isSuccess;
}

unsigned long TensorData::m_convertDenseToSparse(SparseMatrix* sparse,
                                                 const float* dense,
                                                 Shape shape,
                                                 unsigned long paddedRowSize,
                                                 Device device)
{
    throw std::runtime_error("m_convertDenseToSparse not implemented");
}

unsigned long TensorData::m_convertSparseToDense(SparseMatrix* sparse,
                                                 const float* dense,
                                                 Shape shape,
                                                 unsigned long paddedRowSize,
                                                 Device device)
{
    throw std::runtime_error("m_convertSparseToDense not implemented");
}
}  // namespace Motutapu::Util
