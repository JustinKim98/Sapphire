// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TENSOR_DECL_HPP
#define MOTUTAPU_TENSOR_DECL_HPP

#include <Motutapu/tensor/Shape.hpp>
#include <Motutapu/util/Device.hpp>
#include <Motutapu/tensor/TensorDataDecl.hpp>
#include <list>

namespace Motutapu
{
//! TensorData class contains data vector for processing
//! with attributes describing it
template <typename T>
class Tensor
{
public:
    Tensor(Shape shape);

    ~Tensor() = default;

    Tensor(const Tensor<T>& tensor);
    Tensor(Tensor<T>&& tensor) noexcept = delete;
    /// move assignment operator
    Tensor<T>& operator=(const Tensor<T>& tensor);
    Tensor<T>& operator=(Tensor<T>&& tensor) noexcept = delete;

    [[nodiscard]] Shape GetShape() const;
    [[nodiscard]] Device GetDevice() const;

    void RegisterTensorData(Util::TensorData<T>* tensorData);

    Util::TensorData<T>* TensorDataPtr();

private:
    Shape m_shape;

    //! Ptr to the tensorData
    Util::TensorData<T>* m_tensorData = nullptr;

};
}

#endif
