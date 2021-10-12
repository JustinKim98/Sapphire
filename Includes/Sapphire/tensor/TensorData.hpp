// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TENSORUTIL_TENSOR_DATA_HPP
#define SAPPHIRE_TENSORUTIL_TENSOR_DATA_HPP

#include <memory>
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
    TensorData(Shape shape, Type type, CudaDevice device,
               bool preserve = false);

    TensorData(Shape shape, Type type, CudaDevice device,
               int parentDescKey, bool preserve = false);

    //! Shallow copies the internal data
    TensorData(const TensorData& tensorData) = default;
    TensorData(TensorData&& tensorData) noexcept;
    TensorData& operator=(const TensorData& tensorData) = default;
    TensorData& operator=(TensorData&& tensorData) noexcept;
    ~TensorData() = default;

    unsigned long HostTotalSize = 0;
    unsigned long DenseTotalLengthCuda = 0;
    unsigned long SparseTotalLength = 0;

    [[nodiscard]] std::vector<float> GetDataCopy();

    void SetData(std::vector<float> data);

    void Reshape(const Shape& shape);

    [[nodiscard]] int GetDescriptorKey() const
    {
        return m_parentDescKey;
    }

    [[nodiscard]] int GetBatchSize(int requiredDim) const;

    [[nodiscard]] Shape GetShape()
    {
        return m_shape;
    }


    [[nodiscard]] int Rows() const
    {
        return m_shape.Rows();
    }

    [[nodiscard]] int Cols() const
    {
        return m_shape.Cols();
    }

    [[nodiscard]] int Size() const
    {
        return m_shape.Size();
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
        return m_shape;
    }

    [[nodiscard]] std::size_t GetHostElementSize() const
    {
        return m_shape.Size();
    }

    [[nodiscard]] std::size_t GetCudaElementSize() const
    {
        return m_shape.Size();
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
    void ToCuda();

    //! Sends data to host
    //! This operation is available only on Cuda type tensorData
    void ToHost();

    //! Helper static functions
    //! These helper functions are used to control the tensorData from the
    //! operation units

    //! Deep copies tensor data from src to dst
    static void DeepCopy(TensorData& dst, const TensorData& src);


    //!Getters for raw pointers
    [[nodiscard]] const float* HostRawPtr() const
    {
        return m_denseHost;
    }

    [[nodiscard]] const float* CudaRawPtr() const
    {
        return m_denseCuda;
    }

    [[nodiscard]] float* HostMutableRawPtr() const
    {
        return m_denseHost;
    }

    [[nodiscard]] float* CudaMutableRawPtr() const
    {
        return m_denseCuda;
    }


    SparseMatrix* SparseMatHost = nullptr;
    SparseMatrix* SparseMatCuda = nullptr;

private:
    //! Copies data on the Host to Gpu
    //! Only available for Cuda tensors
    void m_toCuda();

    //! Copies data on the Host to Gpu
    //! Only available for Cuda tensors
    void m_toHost();

    //! Allocates data on the Host with given batchSize
    void m_allocateHost();

    //! Allocates data on the GPU with given batchSize
    void m_allocateCuda();

    Shape m_shape;
    float* m_denseHost = nullptr;
    float* m_denseCuda = nullptr;
    int m_parentDescKey = -1;

    Type m_type = Type::Dense;
    DeviceType m_mode = DeviceType::Host;

    CudaDevice m_device;
    bool m_preserve;
};
} // namespace Sapphire::TensorUtil

#endif
