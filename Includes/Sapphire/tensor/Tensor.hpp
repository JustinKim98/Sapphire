// Copyright (c) 2021, Justin Kim
// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TENSOR_DECL_HPP
#define SAPPHIRE_TENSOR_DECL_HPP

#include <Sapphire/util/Shape.hpp>
#include <Sapphire/util/CudaDevice.hpp>
#include <memory>

namespace Sapphire
{
//! TensorDescriptor class contains data vector for processing
//! with attributes describing it
class Tensor
{
public:
    Tensor();
    Tensor(const Shape& shape, const CudaDevice& device, Type type,
           bool preserve = false);

    Tensor(int descKey);
    ~Tensor() = default;

    Tensor(const Tensor& tensor) = default;
    Tensor(Tensor&& tensor) = default;
    /// move assignment operator
    Tensor& operator=(const Tensor& tensor);
    Tensor& operator=(Tensor&& tensor) = default;

    [[nodiscard]] Shape GetShape() const;
    [[nodiscard]] CudaDevice GetDevice() const;
    [[nodiscard]] int TensorDescriptorKey() const;

    void SetDescriptorKey(int key)
    {
        m_tensorDescKey = key;
    }

    [[nodiscard]] std::unique_ptr<float[]> GetForwardDataCopy() const;
    [[nodiscard]] std::unique_ptr<float[]> GetBackwardDataCopy() const;

    void SetForwardData(std::vector<float> data) const;
    void SetBackwardData(const std::vector<float>& data) const;

    void ToCuda();
    void ToHost();
    [[nodiscard]] DeviceType Mode() const;
    void SetMode(DeviceType mode) const;

private:
    int m_tensorDescKey;
};
} // namespace Sapphire

#endif
