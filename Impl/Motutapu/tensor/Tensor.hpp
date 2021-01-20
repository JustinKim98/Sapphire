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
Tensor<T>::Tensor(Shape shape, Device device)
    : m_shape(shape),
      m_initialDevice(device)
{
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor)
    : m_shape(tensor.m_shape),
      m_initialDevice(tensor.m_initialDevice),
      m_tensorData(tensor.m_tensorData),
      m_functionTrajectory(tensor.m_functionTrajectory)
{
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor)
{
    if (&tensor == this)
        return this;

    m_shape = tensor.m_shape;
    m_initialDevice = tensor.m_initialDevice;
    m_tensorData = tensor.m_tensorData;
    m_functionTrajectory = tensor.m_functionTrajectory;

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
    return m_initialDevice;
}

template <typename T>
void Tensor<T>::PushTrajectory(int operationId)
{
    m_functionTrajectory.emplace_back(operationId);
}

template <typename T>
std::optional<int> Tensor<T>::PopTrajectory()
{
    if (m_functionTrajectory.empty())
        return {};

    const int front = m_functionTrajectory.front();
    m_functionTrajectory.pop_front();

    return front;
}

template <typename T>
std::optional<int> Tensor<T>::PeekTrajectory() const
{
    if (m_functionTrajectory.empty())
        return {};

    const int front = m_functionTrajectory.front();
    return front;
}

template <typename T>
std::optional<int> Tensor<T>::GetTensorDataKey() const
{
    if (!m_tensorData)
    {
        return {};
    }

    return m_tensorData->GetKey();
}

template <typename T>
void Tensor<T>::RegisterTensorData(Util::TensorData<T>* tensorData)
{
    m_tensorData = tensorData;
}

}

#endif
