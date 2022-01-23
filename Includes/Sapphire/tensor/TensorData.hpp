// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TENSORUTIL_TENSOR_DATA_HPP
#define SAPPHIRE_TENSORUTIL_TENSOR_DATA_HPP

#include <memory>
#include <vector>
#include <Sapphire/compute/sparse/SparseMatrix.hpp>
#include <Sapphire/util/Shape.hpp>
#include <Sapphire/util/DeviceInfo.hpp>

namespace Sapphire::TensorUtil
{
class TensorData
{
public:
    TensorData() = default;
    //! TensorData is defined only in Host only Mode
    TensorData(Shape shape, Type type,
               bool preserve = false);

    TensorData(Shape shape, Type type, int parentDescKey,
               bool preserve = false);
    //! TensorData is configured in both Host and Cuda Mode
    TensorData(Shape shape, Type type, DeviceInfo device,
               bool preserve = false);

    TensorData(Shape shape, Type type, DeviceInfo device,
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

    //! Gets data stored in this tensorData
    //! \return : vector that contains data copy of current tensorData
    [[nodiscard]] std::vector<float> GetDataCopy();

    //! Sets internal data
    //! Given data will be updated only on cuda if tensorData is in cuda mode,
    //! or it will be updated only on host otherwise
    //! \param data : vector that contains data to load
    void SetData(std::vector<float> data);

    //! Sets cuda device of the tensorData
    //! If TensorData was configured in host mode, tensorData will be able to use
    //! cuda mode after this function is called.
    //! -- TODO --
    //! If TensorData was configured in host & cuda mode, and given device is different
    //! with current device, data will be moved to the new device
    //! \param device : cuda device
    void SetDevice(DeviceInfo device);

    //! Gets current device metadata
    //! This object will be empty if current tensorData is configured in host only mode \return : device descriptor \return : device descriptor
    //! \return : device descriptor
    [[nodiscard]] DeviceInfo GetDeviceInfo() const;

    //! Changes shape of the tensorData
    //! \param shape : shape to change
    void Reshape(const Shape& shape);

    [[nodiscard]] int GetDescriptorKey() const
    {
        return m_parentDescKey;
    }

    //! Get number of unit (batch size) given dimension of required data unit
    //! \param requiredDim : dimension of the data unit
    //! \return : number of units (batch size)
    [[nodiscard]] int GetNumUnits(int requiredDim) const;

    //! Get size of units given dimension of required data unit
    //! \param requiredDim : dimension of the data unit
    //! \return : size of the unit
    [[nodiscard]] int GetUnitSize(int requiredDim) const;

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
    void SetMode(ComputeMode type);

    [[nodiscard]] ComputeMode Mode() const
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
    ComputeMode m_mode = ComputeMode::Host;

    DeviceInfo m_device;
    bool m_preserve;
};
} // namespace Sapphire::TensorUtil

#endif
