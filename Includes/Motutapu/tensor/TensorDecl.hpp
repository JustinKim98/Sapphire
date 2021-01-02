// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_TENSOR_DECL_HPP
#define MOTUTAPU_UTIL_TENSOR_DECL_HPP

#include <Motutapu/util/Shape.hpp>
#include <Motutapu/util/Device.hpp>
#include <vector>
#include <memory>

namespace Motutapu
{
//! TensorData class contains data vector for processing
//! with attributes describing it
template <typename T>
class Tensor
{
 public:
    Tensor() = default;

    Tensor(Shape shape, Device device);

    Tensor(Shape shape, std::size_t batchSize, Compute::Device device);

    Tensor(Shape shape, std::size_t batchSize, Compute::Device device,
           std::vector<T> data);

    ~Tensor();

    Tensor(const Tensor<T>& tensor);
    Tensor(Tensor<T>&& tensor) noexcept = delete;
    /// move assignment operator
    Tensor<T>& operator=(const Tensor<T>& tensor);
    Tensor<T>& operator=(Tensor<T>&& tensor) noexcept = delete;

    void SetData(const std::vector<T>& data);

    [[nodiscard]] std::size_t NumMatrix() const;

    [[nodiscard]] Tensor<T> SubTensor(std::initializer_list<int> index);


    void ChangeBatchSize(std::size_t newBatchSize);

    T& At(std::size_t batchIdx, std::vector<std::size_t> index);

    const T& At(std::size_t batchIdx, std::vector<std::size_t> index) const;

    //! Access the data linearly considering paddings
    T& At(std::size_t idx);

    const T& At(std::size_t idx) const;

    [[nodiscard]] std::size_t ColumnElementSize() const
    {
        return m_columnElementSize;
    }

    [[nodiscard]] std::size_t ElementSize() const
    {
        return m_elementSize;
    }

    [[nodiscard]] std::size_t TotalElementSize() const
    {
        return m_elementSize * BatchSize;
    }

    [[nodiscard]] std::size_t GetDataByteSize() const
    {
        return TotalElementSize() * sizeof(T);
    }

    //! If both tensors are on same device, data is moved rather than copied
    static void ForwardTensorData(Tensor<T>& source, Tensor<T>& destination);

    static void MoveTensorData(Tensor<T>& source, Tensor<T>& destination);

    static void CopyTensorData(const Tensor<T>& source, Tensor<T>& destination);

    static void CopyCPUToGPU(Tensor<T>& dest, Tensor<T>& src);

    static void CopyGPUToCPU(Tensor<T>& dest, Tensor<T>& src);

    static void Sparsify(Tensor<T>& tensor);

    static Tensor<T> Sparesify(const Tensor<T>& tensor);

    /// Shape of this tensorData
    Util::Shape TensorShape;
    Device Device;
    std::size_t BatchSize = 0;

 private:
    std::size_t m_elementSize = 0;
    std::size_t m_columnElementSize = 0;

    std::size_t m_getElementSize() const;

    std::size_t m_getPaddedColumnSize() const;

    void m_freeData();
};

}

#endif