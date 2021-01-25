// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TENSOR_DECL_HPP
#define MOTUTAPU_TENSOR_DECL_HPP

#include <Motutapu/tensor/Shape.hpp>
#include <Motutapu/util/Device.hpp>
#include <Motutapu/tensor/TensorDescriptor.hpp>

#include <list>

namespace Motutapu
{
//! TensorDescriptor class contains data vector for processing
//! with attributes describing it
template <typename T>
class Tensor
{
public:
    Tensor(Shape shape, int descKey);
    ~Tensor() = default;

    Tensor(const Tensor<T>& tensor);
    Tensor(Tensor<T>&& tensor) noexcept = delete;
    /// move assignment operator
    Tensor<T>& operator=(const Tensor<T>& tensor);
    Tensor<T>& operator=(Tensor<T>&& tensor) noexcept = delete;

    [[nodiscard]] Shape GetShape() const;
    [[nodiscard]] Device GetDevice() const;
    [[nodiscard]] int TensorDescriptorKey() const;

private:
    Shape m_shape;

    int m_tensorDescriptorKey;
};
}

#endif
