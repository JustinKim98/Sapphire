// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TENSORUTIL_TENSOR_DATA_HPP
#define SAPPHIRE_TENSORUTIL_TENSOR_DATA_HPP

#include <Sapphire/compute/sparse/SparseMatrix.hpp>
#include <Sapphire/util/Shape.hpp>
#include <Sapphire/util/CudaDevice.hpp>

namespace Sapphire::TensorUtil
{
class TensorData
{
public:
    TensorData() = default;
    //! TensorData is defined only in Host Mode
    TensorData(Shape shape, Type type, bool preserve = false);
    //! TensorData is configured in both Host and Cuda Mode
    TensorData(Shape shape, Type type, CudaDevice device, bool preserve = false);

    TensorData(Shape shape, Type type, CudaDevice device,
               int parentDescKey, bool preserve = false);

    //! Shallow copies the internal data
    TensorData(const TensorData& tensorData);
    TensorData(TensorData&& tensorData) noexcept;
    TensorData& operator=(const TensorData& tensorData);
    TensorData& operator=(TensorData&& tensorData) noexcept;
    ~TensorData();

    unsigned long DenseTotalLengthHost = 0;
    unsigned long DenseTotalLengthCuda = 0;
    unsigned long SparseTotalLength = 0;
    unsigned long PaddedHostColSize = 0;

    [[nodiscard]] const float* GetDenseHost() const
    {
        return DenseMatHost;
    }

    [[nodiscard]] const float* GetDenseCuda() const
    {
        return DenseMatCuda;
    }

    [[nodiscard]] float* GetMutableDenseHost()
    {
        return DenseMatHost;
    }

    [[nodiscard]] float* GetMutableDenseCuda()
    {
        return DenseMatCuda;
    }

    [[nodiscard]] int GetDescriptorKey() const
    {
        return m_parentDescKey;
    }

    [[nodiscard]] int GetBatchSize(int requiredDim) const;

    SparseMatrix* SparseMatHost = nullptr;
    SparseMatrix* SparseMatCuda = nullptr;
    Shape TensorShape;

    [[nodiscard]] int Rows() const
    {
        return TensorShape.Rows();
    }

    [[nodiscard]] int Cols() const
    {
        return TensorShape.Cols();
    }

    //! Gets device descriptor (Sparse or Dense)
    //! \return : device descriptor
    [[nodiscard]] CudaDevice GetDevice() const
    {
        if (Mode() == DeviceType::Host)
            return CudaDevice();
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
        return static_cast<std::size_t>(TensorShape.Size() / TensorShape.Cols())
               * PaddedHostColSize;
    }

    [[nodiscard]] std::size_t GetCudaElementSize() const
    {
        return TensorShape.Size();
    }


    //! Creates and returns same copy as this tensorData
    [[nodiscard]] TensorData CreateCopy() const;


    //! Sets whether cuda or host will execute operations
    //! This operation is available only on Cuda type tensorData
    void SetMode(DeviceType type);

    [[nodiscard]] DeviceType Mode() const
    {
        return m_mode;
    }

    //! Transfers data to target cuda device from current device
    //! immediately returns false if change device is requested to same device
    //! This operation is available only on Cuda type tensorData
    void ToCuda() const;

    //! Sends data to host
    //! This operation is available only on Cuda type tensorData
    void ToHost() const;

    //! Helper static functions
    //! These helper functions are used to control the tensorData from the
    //! operation units

    //! Deep copies tensor data from src to dst
    static void DeepCopy(TensorData& dst, const TensorData& src);

private:
    //! Copies data on the Host to Gpu
    //! Only available for Cuda tensors
    static void m_toCuda(const TensorData& tensorData);

    //! Copies data on the Host to Gpu
    //! Only available for Cuda tensors
    static void m_toHost(const TensorData& tensorData);

    //! Allocates data on the Host with given batchSize
    void m_allocateHost(bool preserve);

    //! Allocates data on the GPU with given batchSize
    void m_allocateCuda(bool preserve);

    //! Free space allocated on Host memory
    void m_freeHost();

    //! Free space allocated on GPU memory
    void m_freeCuda();

    float* DenseMatHost = nullptr;
    float* DenseMatCuda = nullptr;

    int m_parentDescKey = -1;

    Type m_type = Type::Dense;
    DeviceType m_mode = DeviceType::Host;

    CudaDevice m_device;
};
} // namespace Sapphire::TensorUtil

#endif
