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
#include <shared_mutex>

namespace Motutapu::Util
{
enum class Layout
{
    rowMajor,
    columnMajor,
};

template <typename T>
class TensorData
{
public:
    unsigned long DenseTotalLength = 0;
    unsigned long SparseTotalLength = 0;
    unsigned long PaddedColumnSize = 0;
    unsigned long PaddedRowSize = 0;
    unsigned long BatchSize = 0;

    T* DenseMatHost = nullptr;
    T* DenseMatCuda = nullptr;

    SparseMatrix<T>* SparseMatHost = nullptr;
    SparseMatrix<T>* SparseMatCuda = nullptr;
    Shape TensorShape;

    bool IsSparse = false; // True if Dense, False if Sparse
    std::atomic<bool> IsBusy = false;

    Device CurDevice;

    void SetKey(int key);

    [[nodiscard]] int GetKey();

    static TensorData<T>* CreateTensorData(
        Shape shape, Device device,
        bool isSparse, size_t batchSize);

    static bool DestroyTensorData(TensorData<T>* tensorData);

    static void DenseToSparse(TensorData<T>* tensorData, Device device);
    static void SparseToDense(TensorData<T>* tensorData, Device device);

    static bool CopyTensorData(TensorData<T>* dest, const TensorData<T>* src,
                               Device device);

    static void CopyHostToGpu(TensorData<T>* tensorData);
    static void CopyGpuToHost(TensorData<T>* tensorData);

private:
    void m_allocate(unsigned long batchSize);

    static unsigned long m_convertDenseToSparse(SparseMatrix<T>* sparse,
                                                const T* dense,
                                                Shape shape,
                                                unsigned long paddedRowSize,
                                                Device device);

    static unsigned long m_convertSparseToDense(SparseMatrix<T>* sparse,
                                                const T* dense, Shape shape,
                                                unsigned long paddedRowSize,
                                                Device device);

    TensorData(Shape shape, bool isSparse, Device device);
    ~TensorData() = default;

    //! Key to identify tensor data
    int m_key;

    //! mutex to make sure operations on the resources is synchronized
    std::shared_mutex m_mtx;
};
} // namespace Motutapu::Util

#endif
