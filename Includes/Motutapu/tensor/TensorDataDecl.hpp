// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_TENSORDATA_DECL_HPP
#define MOTUTAPU_UTIL_TENSORDATA_DECL_HPP

#include <Motutapu/util/Device.hpp>
#include <Motutapu/util/SparseMatrix.hpp>
#include <Motutapu/tensor/Shape.hpp>
#include <list>
#include <mutex>

namespace Motutapu::Util
{
//! This describes history of the tensorData
//! As tensorData is used in unit function as an operand or input/output.
//! It is stored using this struct
struct History
{
    History(int unitKey)
        : UnitKey(unitKey)
    {
    }

    History() = default;

    bool IsOutput();

    //! Key of the unit
    int UnitKey = -1;
    //! List of the units that was as operand
    std::list<int> OperandUnitKeyList;
};

//! TensorData stores real tensor data of the tensor
//! There can be more than one tensor that references to one tensorData
//! All public functions in the TensorData is guaranteed to be thread safe
//! TensorData should not be accessible from the user interface directly
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

    //! Key to identify tensor data
    int Key = -1;

    //! Gets device descriptor (Sparse or Dense)
    //! \return : device descriptor
    [[nodiscard]] const Device& GetDevice() const
    {
        return m_device;
    }

    //! Gets type of the data (Sparse of Dense)
    //! \return : Type of the data
    [[nodiscard]] Type GetType() const
    {
        return m_type;
    }

    //! Add unit Key if unit was used as output or flow-through type
    //! \param unitKey : unitKey to add
    void AddOutputUnitHistory(int unitKey);

    //! Add unit key if unit was used as operand only
    //! \param unitKey : unitKey to add
    void AddOperandUnitHistory(int unitKey);

    //! Accept the gradient from the operand unit
    //! \param unitKey : Key of the unit to accept gradient from
    void AcceptGrad(int unitKey);

    //! Helper static functions
    //! These helper functions are used to control the tensorData from the
    //! operation units

    //! Creates tensor data and allocates the space
    //! throws runtime error if creation process was not successful
    //! \param shape : shape of the tensor data to create
    //! \param device : initial device to allocate tensorData
    //! \param type : Type of the tensor data (Dense or Sparse)
    //! \param batchSize : batch size of the tensor data
    static TensorData<T>* CreateTensorData(
        const Shape& shape, const Device& device,
        Type type, unsigned batchSize);

    //! Frees and invalidates tensor data
    //! \param tensorData : tensorData to deallocate
    //! \return : true if destruction was successful false otherwise
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

    //! Checks if next operation is output unit in back propagation
    //! \return : true if ready false otherwise
    bool IsBackPropReady()
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);

        if (m_history.empty())
            return false;

        if (m_history.back().IsOutput())
            return true;

        return false;
    }

private:

    //! Allocates data on the CPU with given batchSize
    void m_allocateCpu(unsigned int batchSize);

    //! Allocates data on the GPU with given batchSize
    bool m_allocateCuda(unsigned int batchSize);

    //! Free space allocated on CPU memory
    void m_freeCpu() const;

    //! Free space allocated on GPU memory
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

    ~TensorData() = default;

    std::list<History> m_history;

    Type m_type;
    Device m_device;
    //! mutex to make sure operations on the resources is synchronized
    std::recursive_mutex m_mtx;
};
} // namespace Motutapu::Util

#endif
