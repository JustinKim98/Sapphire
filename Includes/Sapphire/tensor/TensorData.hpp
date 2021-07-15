// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_UTIL_TENSORDATA_DECL_HPP
#define Sapphire_UTIL_TENSORDATA_DECL_HPP

#include <Sapphire/compute/sparse/SparseMatrix.hpp>
#include <Sapphire/tensor/Shape.hpp>
#include <Sapphire/util/Device.hpp>

namespace Sapphire::TensorUtil
{
class TensorData
{
public:
    TensorData() = default;
    TensorData(Shape shape, Type type, Device device, unsigned int batchSize);

    TensorData(Shape shape, Type type, Device device, unsigned int batchSize,
               int parentDescKey);

    TensorData(const TensorData& tensorData);
    TensorData(TensorData&& tensorData) noexcept;
    TensorData& operator=(const TensorData& tensorData);
    TensorData& operator=(TensorData&& tensorData) noexcept;
    ~TensorData();

    unsigned long DenseTotalLengthHost = 0;
    unsigned long DenseTotalLengthCuda = 0;
    unsigned long SparseTotalLength = 0;
    unsigned long PaddedHostColSize = 0;
    unsigned long BatchSize = 0;

    float* DenseMatHost = nullptr;
    float* DenseMatCuda = nullptr;

    [[nodiscard]] int GetDescriptorKey() const
    {
        return m_parentDescKey;
    }

    SparseMatrix* SparseMatHost = nullptr;
    SparseMatrix* SparseMatCuda = nullptr;
    Shape TensorShape;

    [[nodiscard]] unsigned int Rows() const
    {
        return TensorShape.Rows();
    }

    [[nodiscard]] unsigned int Cols() const
    {
        return TensorShape.Cols();
    }

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

    [[nodiscard]] Shape GetShape() const
    {
        return TensorShape;
    }

    [[nodiscard]] std::size_t GetHostElementSize() const
    {
        return (TensorShape.Size() / TensorShape.Cols()) * PaddedHostColSize;
    }

    [[nodiscard]] std::size_t GetCudaElementSize() const
    {
        return TensorShape.Size();
    }

    //! Helper static functions
    //! These helper functions are used to control the tensorData from the
    //! operation units

    //! Deep copies tensor data from src to dest
    //! Type of dest and src must be the same
    static void CopyTensorData(TensorData dest, const TensorData& src);

    //! Creates and returns same copy as this tensorData
    [[nodiscard]] TensorData CreateCopy() const;

    //! Changes device of the tensor
    //! Transfers data to target device from current device
    //! immediately returns false if change device is requested to same device
    //! \param device : new device to set
    bool SendTo(const Device& device);

    //! Deep copies tensor data from src to dst
    static void DeepCopy(TensorData& dst, const TensorData& src);

private:
    //! Copies data on the Host to Gpu
    //! Only available for CUDA tensors
    static void m_toGpu(const TensorData& tensorData);

    //! Copies data on the Host to Gpu
    //! Only available for CUDA tensors
    static void m_toHost(const TensorData& tensorData);

    //! Allocates data on the HOST with given batchSize
    void m_allocateHost(unsigned int batchSize);

    //! Allocates data on the GPU with given batchSize
    void m_allocateCuda(unsigned int batchSize);

    //! Free space allocated on HOST memory
    void m_freeHost();

    //! Free space allocated on GPU memory
    void m_freeCuda();

    int m_parentDescKey = -1;

    Type m_type = Type::Dense;

    Device m_device;
};
} // namespace Sapphire::TensorUtil

#endif
