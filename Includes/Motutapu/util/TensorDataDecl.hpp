// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TENSORDATA_DECL_HPP
#define MOTUTAPU_TENSORDATA_DECL_HPP

#include <Motutapu/util/Device.hpp>
#include <Motutapu/util/SparseMatrix.hpp>
#include <Motutapu/tensor/Shape.hpp>
#include <atomic>

namespace Motutapu::Util
{
template <typename T>
class TensorData
{
public:
    unsigned long DenseTotalLength = 0;
    unsigned long SparseTotalLength = 0;
    unsigned long PaddedRowSize = 0;
    unsigned long BatchSize = 0;

    T* DenseMatHost = nullptr;
    T* DenseMatCuda = nullptr;
    SparseMatrix<T>* SparseMatHost = nullptr;
    SparseMatrix<T>* SparseMatCuda = nullptr;
    Shape TensorShape;

    bool IsSparse = false; // True if Dense, False if Sparse
    std::atomic<bool> IsBusy = false;

    static TensorData<T>* CreateTensorData(unsigned long batchSize, Shape shape,
                                           bool isSparse, Device device);
    static bool DestroyTensorData(TensorData<T>* tensorData, Device device);

    static void DenseToSparse(TensorData<T>* tensorData, Device device);
    static void SparseToDense(TensorData<T>* tensorData, Device device);

    static bool CopyTensorData(TensorData<T>* dest, const TensorData<T>* src,
                               Device device);

    static void CopyHostToGpu(TensorData<T>* tensorData);
    static void CopyGpuToHost(TensorData<T>* tensorData);

private:

    static unsigned long m_convertDenseToSparse(SparseMatrix<T>* sparse,
                                              const T* dense,
                                              Shape shape,
                                              unsigned long paddedRowSize,
                                              Device device);

    static unsigned long m_convertSparseToDense(SparseMatrix<T>* sparse,
                                              const T* dense, Shape shape,
                                              unsigned long paddedRowSize,
                                              Device device);

    TensorData(unsigned long batchSize, Shape shape, bool isSparse);
    ~TensorData();
};
} // namespace Motutapu::Util

#endif
