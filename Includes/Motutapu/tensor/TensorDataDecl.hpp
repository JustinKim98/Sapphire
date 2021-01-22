// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_TENSORDATA_DECL_HPP
#define MOTUTAPU_UTIL_TENSORDATA_DECL_HPP

#include <Motutapu/util/Device.hpp>
#include <Motutapu/util/SparseMatrix.hpp>
#include <Motutapu/tensor/Shape.hpp>
#include <atomic>
#include <shared_mutex>

namespace Motutapu::Util
{
enum class Type
{
    Sparse,
    Dense,
};

template <typename T>
class TensorData
{
public:

    TensorData(const TensorData& tensorData) = delete;
    TensorData(TensorData&& tensorData) noexcept = default;
    TensorData& operator=(const TensorData& tensorData) = delete;
    TensorData& operator=(TensorData&& tensorData) noexcept = default;

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

    std::atomic<bool> IsBusy = false;


    //! Gets device descriptor (Sparse or Dense)
    //! \return : device descriptor
    [[nodiscard]] Device GetDevice() const
    {
        return m_device;
    }

    //! Gets type of the data (Sparse of Dense)
    //! \return : Type of the data
    [[nodiscard]] Type GetType() const
    {
        return m_type;
    }

    //! Gets key that represents this tensorData
    //! \return : Key of the tensorData
    [[nodiscard]] int GetKey() const
    {
        return m_key;
    }

    //! Sets key that represents this tensorData
    //! \param key : key to set
    void SetKey(int key)
    {
        m_key = key;
    }

    //! Creates tensor data and allocates the space
    //! \param shape : shape of the tensor data to create
    //! \param device : initial device to allocate tensorData
    //! \param type : Type of the tensor data (Dense or Sparse)
    //! \param batchSize : batch size of the tensor data
    static TensorData<T>* CreateTensorData(
        const Shape& shape, const Device& device,
        Type type, unsigned batchSize);

    //! Frees and invalidates tensor data
    //! \param tensorData : tensorData to deallocate
    static bool DestroyTensorData(TensorData<T>* tensorData);

    //! Converts tensor data from dense to sparse
    static void DenseToSparse(TensorData<T>* tensorData);
    //! Converts tensor data  from sparse to dense
    static void SparseToDense(TensorData<T>* tensorData);

    //! Deep copies tensor data from src to dest
    //! Type of dest and src must be the same
    static bool CopyTensorData(TensorData<T>* dest, const TensorData<T>* src);

    //! Changes device of the tensor
    //! Transfers data to target device from current device
    //! immediately returns false if change device is requested to same device
    //! \param tensorData : tensorData object to change device
    //! \param device : new device to set
    static bool ChangeDevice(TensorData<T>* tensorData, Device device);

    //! Copies data on the Host to Gpu
    //! Only available for CUDA tensors
    static void CopyHostToGpu(TensorData<T>* tensorData);
    //! Copies data on the Host to Gpu
    //! Only available for CUDA tensors
    static void CopyGpuToHost(TensorData<T>* tensorData);

private:
    void m_allocateCpu(unsigned int batchSize);
    bool m_allocateCuda(unsigned int batchSize);

    void m_freeCpu() const;
    bool m_freeGpu();


    static unsigned long m_convertDenseToSparse(SparseMatrix<T>* sparse,
                                                const T* dense,
                                                Shape shape,
                                                unsigned long paddedRowSize,
                                                Device device);

    static unsigned long m_convertSparseToDense(SparseMatrix<T>* sparse,
                                                const T* dense, Shape shape,
                                                unsigned long paddedRowSize,
                                                Device device);

    TensorData(Shape shape, Type type, Device device);

private:
    ~TensorData() = default;

    //! Key to identify tensor data
    int m_key = -1;
    Type m_type;
    Device m_device;
    //! mutex to make sure operations on the resources is synchronized
    std::shared_mutex m_mtx;
};
} // namespace Motutapu::Util

#endif
