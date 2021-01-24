// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_TENSORDATA_HPP
#define MOTUTAPU_UTIL_TENSORDATA_HPP

#include <Motutapu/tensor/TensorDataDecl.hpp>
#include <Motutapu/compute/cuda/Memory.cuh>
#include <mutex>
#include <algorithm>
#include <memory>
#include <type_traits>

namespace Motutapu::Util
{
template <typename T>
void TensorData<T>::AddOutputUnitHistory(int unitKey)
{
    m_history.emplace_back(History(unitKey));
}

template <typename T>
void TensorData<T>::AddOperandUnitHistory(int unitKey)
{
    if (m_history.back().IsOutput())
    {
        m_history.emplace_back(History());
    }
    else
    {
        m_history.back().OperandUnitKeyList.emplace_back(unitKey);
    }
}

template <typename T>
void TensorData<T>::AcceptGrad(int unitKey)
{
    std::lock_guard<std::recursive_mutex> lock(m_mtx);

    if (m_history.empty())
    {
        throw std::runtime_error("AcceptGrad - History is empty");
    }
    if (m_history.back().IsOutput())
    {
        throw std::runtime_error("AcceptGrad - Last history was output");
    }

    History& hist = m_history.back();
    const auto resultItr = std::find(hist.OperandUnitKeyList.begin(),
                                     hist.OperandUnitKeyList.end(), unitKey);

    //! Received gradient should be marked in the history
    if (resultItr == hist.OperandUnitKeyList.end())
    {
        throw std::runtime_error(
            "AcceptGrad - requested Operand history was not found");
    }

    hist.OperandUnitKeyList.erase(resultItr);

    //! Remove last history if OperandUnitKeyList was filled
    if (hist.OperandUnitKeyList.empty())
    {
        m_history.pop_back();
    }
}

template <typename T>
TensorData<T>* TensorData<T>::CreateTensorData(const Shape& shape,
                                               const Device& device,
                                               Type type, unsigned batchSize)
{
    auto success = true;
    auto* tensorData = new TensorData<T>(shape, type, device);
    if (device.Type() == DeviceType::CUDA)
    {
        success &= Compute::Cuda::CudaSetDevice(device.GetID());
        success &= tensorData->m_allocateCuda(batchSize);
        tensorData->m_allocateCpu(batchSize);
    }

    if (!success)
    {
        throw std::runtime_error("Tensor creation failed");
    }

    return tensorData;
}

template <typename T>
bool TensorData<T>::DestroyTensorData(TensorData<T>* tensorData)
{
    std::unique_lock<std::recursive_mutex> lock(tensorData->m_mtx);
    auto isSuccess = true;

    tensorData->m_freeCpu();

    if (tensorData->m_device.Type() == DeviceType::CUDA)
    {
        isSuccess &= tensorData->m_freeGpu();
    }

    lock.release();

    delete tensorData;

    return isSuccess;
}

template <typename T>
void TensorData<T>::DenseToSparse(TensorData<T>* tensorData)
{
    throw std::runtime_error("DenseToSparse not implemented");
}

template <typename T>
void TensorData<T>::SparseToDense(TensorData<T>* tensorData)
{
    throw std::runtime_error("SparseToDense not implemented");
}

template <typename T>
bool TensorData<T>::CopyTensorData(TensorData<T>* dest,
                                   const TensorData<T>* src)
{
    std::unique_lock<std::recursive_mutex> src_lock(src->m_mtx);
    std::unique_lock<std::recursive_mutex> dest_lock(dest->m_mtx);

    if (src->GetDevice() != dest->GetDevice())
    {
        throw std::invalid_argument("Device mismatch while copying tensorData");
    }

    if (dest->TensorShape != src->TensorShape)
    {
        throw std::invalid_argument("Shape mismatch while copying tensorData");
    }

    if (dest->GetType() != src->GetType())
    {
        throw std::invalid_argument("Type mismatch while copying tensorData");
    }

    const bool sparse = src->GetType() == Type::Sparse ? true : false;
    bool success = true;
    auto device = src->GetDevice();

    const Shape shape = dest->TensorShape;

    if (device.Type() == DeviceType::CPU)
    {
        if (sparse)
        {
            throw std::runtime_error("CopyTensorData - sparse not implemented");
        }
        else
        {
            std::memcpy(dest->DenseMatHost, src->DenseMatHost,
                        src->DenseTotalLength * sizeof(T));
            dest->DenseTotalLength = src->DenseTotalLength;
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
            Compute::Cuda::MemcpyGpuToGpuFloat(
                dest->DenseMatCuda,
                src->DenseMatCuda,
                src->DenseTotalLength);
            dest->DenseTotalLength = src->DenseTotalLength;
        }
    }

    dest_lock.release();
    src_lock.release();

    return success;
}

template <typename T>
bool TensorData<T>::ChangeDevice(TensorData<T>* tensorData, Device device)
{
    static_assert(
        std::disjunction_v<std::is_same<T, half>, std::is_same<T, float>>,
        "CopyHostToGpu - Unsupported data type for GPU");

    std::lock_guard<std::recursive_mutex> lock(tensorData->m_mtx);
    auto currentDevice = tensorData->GetDevice();
    if (currentDevice == device)
    {
        return false;
    }

    if (currentDevice.Type() == DeviceType::CPU && device.Type() ==
        DeviceType::CUDA)
    {
        tensorData->m_allocateCuda(tensorData->BatchSize);
        TensorData<T>::CopyHostToGpu(tensorData);
    }

    if (currentDevice.Type() == DeviceType::CUDA && device.Type() ==
        DeviceType::CPU)
    {
        TensorData<T>::CopyGpuToHost(tensorData);
    }

    if (currentDevice.Type() == DeviceType::CUDA &&
        device.Type() == DeviceType::CUDA)
    {
        TensorData<T>::CopyGpuToHost(tensorData);
        TensorData<T>::CopyHostToGpu(tensorData);
    }

    tensorData->m_device = device;

    return true;
}

template <typename T>
void TensorData<T>::CopyHostToGpu(TensorData<T>* tensorData)
{
    static_assert(
        std::disjunction_v<std::is_same<T, half>, std::is_same<T, float>>,
        "CopyHostToGpu - Unsupported data type for GPU");

    std::lock_guard<std::recursive_mutex> lock(tensorData->m_mtx);
    if (tensorData->GetDevice().Type() != DeviceType::CUDA)
    {
        throw std::invalid_argument(
            "CopyHostToGpu - Given tensor data is not GPU tensor");
    }

    if (tensorData->GetType() == Type::Sparse)
    {
        throw std::runtime_error("Sparse matrix not implemented");
    }
    else
    {
        Compute::Cuda::CudaSetDevice(tensorData->m_device.GetID());
        if constexpr (std::is_same_v<T, float>)
            Compute::Cuda::MemcpyHostToGpuFloat(tensorData->DenseMatCuda,
                                                tensorData->DenseMatHost,
                                                tensorData->BatchSize);
        else if constexpr (std::is_same_v<T, half>)
            Compute::Cuda::MemcpyHostToGpuHalf(tensorData->DenseMatCuda,
                                               tensorData->DenseMatHost,
                                               tensorData->BatchSize);
    }
}

template <typename T>
void TensorData<T>::CopyGpuToHost(TensorData<T>* tensorData)
{
    static_assert(
        std::disjunction_v<std::is_same<T, half>, std::is_same<T, float>>,
        "CopyHostToGpu - Unsupported data type for GPU");

    std::lock_guard<std::recursive_mutex> lock(tensorData->m_mtx);
    if (tensorData->GetDevice().Type() != DeviceType::CUDA)
    {
        throw std::invalid_argument(
            "CopyHostToGpu - Given tensor data is not GPU tensor");
    }

    if (tensorData->GetType() == Type::Sparse)
    {
        throw std::runtime_error("Sparse matrix not implemented");
    }
    else
    {
        Compute::Cuda::CudaSetDevice(tensorData->m_device.GetID());

        if constexpr (std::is_same_v<T, float>)
            Compute::Cuda::MemcpyGpuToHostFloat(tensorData->DenseMatHost,
                                                tensorData->DenseMatCuda,
                                                tensorData->BatchSize);
        else if constexpr (std::is_same_v<T, half>)
            Compute::Cuda::MemcpyGpuToHostHalf(tensorData->DenseMatHost,
                                               tensorData->DenseMatCuda,
                                               tensorData->BatchSize);
    }
}

template <typename T>
void TensorData<T>::m_freeCpu() const
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

template <typename T>
bool TensorData<T>::m_freeGpu()
{
    bool isSuccess = true;

    isSuccess &= Compute::Cuda::CudaSetDevice(m_device.GetID());

    if (m_type == Type::Sparse)
    {
        isSuccess &= Compute::Cuda::CudaFree(
            reinterpret_cast<void**>(SparseMatCuda));
    }
    else
    {
        isSuccess &= Compute::Cuda::CudaFree(
            reinterpret_cast<void**>(DenseMatCuda));
    }

    return isSuccess;
}

template <typename T>
void TensorData<T>::m_allocateCpu(unsigned int batchSize)
{
    const auto colSize = TensorShape.At(0);
    const auto rowSize = TensorShape.Dim() > 1 ? TensorShape.At(1) : 0;

    const auto padUnitSize = static_cast<unsigned int>(32 / sizeof(T));

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
        DenseMatHost = new T[DenseTotalLength];
    }
}

template <typename T>
bool TensorData<T>::m_allocateCuda(unsigned int batchSize)
{
    static_assert(
        std::disjunction_v<std::is_same<T, half>, std::is_same<T, float>>,
        "CopyHostToGpu - Unsupported data type for GPU");

    const auto colSize = TensorShape.At(0);
    const auto rowSize = TensorShape.Dim() > 1 ? TensorShape.At(1) : 0;

    const unsigned int padUnitSize = 32 / sizeof(T);

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
            if constexpr (std::is_same_v<T, float>)
                isSuccess &= Compute::Cuda::CudaMallocFloat(
                    &DenseMatCuda, batchSize * paddedRowSize * paddedColSize);
            else if constexpr (std::is_same_v<T, half>)
                isSuccess &= Compute::Cuda::CudaMallocHalf(
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


template <typename T>
unsigned long TensorData<T>::m_convertDenseToSparse(
    SparseMatrix<T>* sparse, const T* dense,
    Shape shape, unsigned long paddedRowSize,
    Device device)
{
    throw std::runtime_error("m_convertDenseToSparse not implemented");
}

template <typename T>
unsigned long TensorData<T>::m_convertSparseToDense(
    SparseMatrix<T>* sparse, const T* dense,
    Shape shape, unsigned long paddedRowSize,
    Device device)
{
    throw std::runtime_error("m_convertSparseToDense not implemented");
}

template <typename T>
TensorData<T>::TensorData(Shape shape, Type type, Device device)
    : TensorShape(std::move(shape)),
      m_type(type),
      m_device(std::move(device))
{
}
} // namespace Motutapu::Util

#endif
