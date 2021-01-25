// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TENSOR_HPP
#define MOTUTAPU_TENSOR_HPP
#include <Motutapu/tensor/TensorDecl.hpp>
#include <Motutapu/Model.hpp>

namespace Motutapu
{
template <typename T>
Tensor<T>::Tensor(Shape shape, int descKey)
    : m_shape(shape),
      m_tensorDescriptorKey(descKey)
{
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor)
    : m_shape(tensor.m_shape),
      m_tensorDescriptorKey(tensor.m_tensorDescriptorKey)
{
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor)
{
    if (&tensor == this)
        return this;

    m_shape = tensor.m_shape;
    m_tensorDescriptorKey = tensor.m_tensorDescriptorKey;
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
    Model& model = ModelManager::GetCurrentModel();
    Util::TensorDescriptor<T>& desc = model.GetDescriptor<T>(
        m_tensorDescriptorKey);
    return desc.ForwardData.GetDevice();
}

template <typename T>
int Tensor<T>::TensorDescriptorKey() const
{
    return m_tensorDescriptorKey;
}
}

#endif
