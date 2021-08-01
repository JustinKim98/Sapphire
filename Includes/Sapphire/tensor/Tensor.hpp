// Copyright (c) 2021, Justin Kim
// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TENSOR_DECL_HPP
#define SAPPHIRE_TENSOR_DECL_HPP

#include <Sapphire/tensor/Shape.hpp>
#include <Sapphire/util/Device.hpp>

namespace Sapphire
{
//! TensorDescriptor class contains data vector for processing
//! with attributes describing it
class Tensor
{
public:
    Tensor(const Shape& shape, unsigned int batchSize, const Device& device,
           Type type = Type::Dense);
    Tensor(int descKey);
    ~Tensor() = default;

    Tensor(const Tensor& tensor) = default;
    Tensor(Tensor&& tensor) noexcept = delete;
    /// move assignment operator
    Tensor& operator=(const Tensor& tensor);
    Tensor& operator=(Tensor&& tensor) noexcept = delete;

    [[nodiscard]] Shape GetForwardDataShape() const;
    [[nodiscard]] Device GetDevice() const;
    [[nodiscard]] int TensorDescriptorKey() const;

    //! Set Tensor device
    void SendTo(const Device& device) const;

private:
    int m_tensorDescKey;
};
} // namespace Sapphire

#endif
