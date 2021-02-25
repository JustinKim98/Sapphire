// Copyright (c) 2021, Justin Kim
// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TENSOR_DECL_HPP
#define MOTUTAPU_TENSOR_DECL_HPP

#include <Motutapu/tensor/Shape.hpp>
#include <Motutapu/util/Device.hpp>

namespace Motutapu
{


//! TensorDescriptor class contains data vector for processing
//! with attributes describing it
class Tensor
{
 public:
    Tensor(Shape shape, unsigned int descKey);
    ~Tensor() = default;

    Tensor(const Tensor& tensor) = default;
    Tensor(Tensor&& tensor) noexcept = delete;
    /// move assignment operator
    Tensor& operator=(const Tensor& tensor);
    Tensor& operator=(Tensor&& tensor) noexcept = delete;

    [[nodiscard]] Shape GetShape() const;
    [[nodiscard]] Device GetDevice() const;
    [[nodiscard]] int TensorDescriptorKey() const;

    //! Set Tensor device
    void SendTo(const Device& device) const;

 private:
    Shape m_shape;
    int m_tensorDescriptorKey;
};
}  // namespace Motutapu

#endif
