// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TENSOR_HPP
#define MOTUTAPU_TENSOR_HPP
#include <optional>
#include <Motutapu/tensor/TensorDecl.hpp>

namespace Motutapu
{
template <typename T>
Tensor<T>::Tensor(Shape shape)
    : m_shape(shape)
{
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor)
    : m_shape(tensor.m_shape),
      m_tensorData(tensor.m_tensorData)
{
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor)
{
    if (&tensor == this)
        return this;

    m_shape = tensor.m_shape;
    m_tensorData = tensor.m_tensorData;

    return this;
}

template <typename T>
Shape Tensor<T>::GetShape() const
{
    return m_shape;
}

template <typename T>
Device Tensor<T>::GetDevice() const
{
    return m_tensorData->GetDevice();
}

template <typename T>
Util::TensorData<T>* Tensor<T>::TensorDataPtr()
{
    return m_tensorData;
}

template <typename T>
void Tensor<T>::RegisterTensorData(Util::TensorData<T>* tensorData)
{
    m_tensorData = tensorData;
}

}

#endif
