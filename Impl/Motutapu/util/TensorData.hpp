// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TENSORDATA_HPP
#define MOTUTAPU_TENSORDATA_HPP

#include <Motutapu/util/TensorDataDecl.hpp>
#include <Motutapu/compute/cuda/Memory.hpp>

namespace Motutapu::Util
{
template <typename T>
TensorData<T>* TensorData<T>::CreateTensorData(unsigned long batchSize,
                                               Shape shape, bool isSparse,
                                               Device device)
{
    auto* tensorData = new TensorData<T>(batchSize, shape, isSparse);

    const size_t size = batchSize * shape.Size();
    tensorData->DenseMatHost = new T[size];

#ifdef WITH_CUDA
    if (device.Type() == DeviceType::CUDA)
    {
        CudaSetDevice(device.Id());
        CudaMalloc(static_cast<void**>(&tensorData->DenseMatCuda),
                   size * sizeof(T));
    }
#endif

    return tensorData;
}

template <typename T>
bool TensorData<T>::DestroyTensorData(TensorData<T>* tensorData, Device device)
{
    bool isSuccess = true;
    if (tensorData->IsSparse)
    {
        delete[] tensorData->SparseMatHost;
    }
    else
    {
        delete[] tensorData->DenseMatHost;
    }

#ifdef WITH_CUDA
    if (device.Type() == DeviceType::CUDA)
    {
        isSuccess &= CudaSetDevice(device.Id());
        if (tensorData->IsSparse)
        {
            isSuccess &=
                CudaFree(static_cast<void**>(tensorData->SparseMatCuda));
        }
        else
        {
            isSuccess &=
                CudaFree(static_cast<void**>(tensorData->DenseMatCuda));
        }
    }
#endif

    delete tensorData;
    return isSuccess;
}

template <typename T>
void TensorData<T>::DenseToSparse(TensorData<T>* tensorData)
{
}

template <typename T>
void TensorData<T>::SparseToDense(TensorData<T>* tensorData)
{
}

template <typename T>
bool TensorData<T>::CopyTensorData(TensorData<T>* dest,
                                   const TensorData<T>* src, Device device)
{
    if (dest->TensorShape != src->TensorShape)
    {
        throw std::invalid_argument("Shape mismatch while copying tensorData");
    }

    if (src->IsBusy || dest->IsBusy)
        return false;

    bool success = true;

    src->IsBusy.exchange(true, std::memory_order_acquire);
    dest->IsBusy.exchange(true, std::memory_order_acquire);

    const Shape shape = dest->TensorShape;

    if (device.Type() == DeviceType::CPU)
    {
        if (dest->IsSparse)
        {
            delete[] dest->SparseMatHost;
            dest->SparseMatHost = new SparseMatrix<T>[src->SparseTotalLength];
            dest->DenseTotalLength = 0;
            if (!src->IsSparse)
            {
                auto length = ConvertDenseToSparse(dest->SparseMatHost,
                                                   src->DenseMatHost,
                                                   shape, src->PaddedRowSize,
                                                   device);
                dest->SparseTotalLength = length;
            }
            else
            {
                //! Find way to correctly copy sparse matrices
                std::memcpy(dest->SparseMatHost, src->SparseMatHost,
                            src->SparseTotalLength * sizeof(T));
                dest->SparseTotalLength = src->SparseTotalLength;
            }
        }
        else
        {
            if (src->IsSparse)
            {
                auto length = ConverSparseToDense(dest->DenseMatHost,
                                                  src->SparseMatHost,
                                                  shape, src->PaddedRowSize,
                                                  device);
                dest->DenseTotalLength = length;
            }
            else
            {
                std::memcpy(dest->SparseMatHost, src->SparseMatHost,
                            src->SparseTotalLength * sizeof(T));
                dest->DenseTotalLength = src->DenseTotalLength;
            }
        }
    }
#ifdef WITH_CUDA

    if (device.Type() == DeviceType::CUDA)
    {
        success &= CudaSetDevice(device.Id());

        if (dest->IsSparse)
        {
            success &= CudaFree(static_cast<void**>(dest->SparseMatCuda));
            success &= CudaMalloc(static_cast<void**>(dest->SparseMatCuda),
                                  src->SparseTotalLength * sizeof(T));
            if (!src->IsSparse)
            {
                auto length = ConvertDenseToSparse(
                    dest->SparseMatCuda, src->DenseMatCuda,
                    shape, 0, device);

                dest->SparseTotalLength = length;
            }
            else
            {
                success &= CopySparseGpuToGpu(dest->SparseMatCuda,
                                              src->DenseMatCuda,
                                              src->BatchSize);
                dest->SparseTotalLength = src->SparseTotalLength;
            }
        }
        else
        {
            if (src->IsSparse)
            {
                auto length = ConvertSparseToDense(dest->DenseMatCuda,
                                                   src->SparseMatCuda,
                                                   shape, dest->PaddedRowSize,
                                                   device);
                dest->DenseTotalLength = length;
            }
            else
            {
                MemcpyGpuToGpu(
                    static_cast<void**>(dest->DenseMatCuda),
                    static_cast<void**>(src->DenseMatCuda),
                    src->DenseTotalLength * sizeof(T));
                dest->DenseTotalLength = src->DenseTotalLength;
            }
        }

#endif

        src->IsBusy.exchange(true, std::memory_order_release);
        dest->IsBusy.exchange(true, std::memory_order_release);
    }

    return success;
}

template
<typename T>
TensorData<T>::TensorData(unsigned long batchSize, Shape shape,
                          bool isSparse)
    : BatchSize(batchSize),
      TensorShape(shape),
      IsSparse(isSparse)
{
}
} // namespace Motutapu::Util

#endif
