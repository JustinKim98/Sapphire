// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_MODEL_HPP
#define MOTUTAPU_MODEL_HPP

#include <Motutapu/operations/ModelDecl.hpp>
#include <type_traits>

namespace Motutapu
{
inline Model::Model(size_t batchSize, std::string name)
    : m_batchSize(batchSize),
      m_name(std::move(name))
{
}

inline Model::~Model()
{
    for (auto [key, tensorData] : m_tensorDataPool.FloatTensorDataMap)
    {
        Util::TensorData<float>::DestroyTensorData(tensorData);
    }

    for (auto [key, tensorData] : m_tensorDataPool.DoubleTensorDataMap)
    {
        Util::TensorData<double>::DestroyTensorData(tensorData);
    }

    for (auto [key, tensorData] : m_tensorDataPool.IntTensorDataMap)
    {
        Util::TensorData<int>::DestroyTensorData(tensorData);
    }
}

template <typename T>
void Model::Register(Unit<T>* unit)
{
    for (const auto& [name, tensor] : unit->OutputTensorMap)
    {
        auto* tensorData = Util::TensorData<T>::CreateTensorData(
            tensor.GetShape(), tensor.GetDevice(), false, m_batchSize);

        tensor.RegisterTensorData(tensorData);
        tensorData->SetKey(m_tensorDataPool.Counter++);
        unit->BatchSize = m_batchSize;

        if constexpr (std::is_same_v<T, float>)
        {
            m_unitPool.FloatUnitMap[m_unitPool.Counter++] = unit;
            m_tensorDataPool.FloatTensorDataMap[m_tensorDataPool.Counter++] =
                tensorData;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            m_unitPool.DoubleUnitMap[m_unitPool.Counter++] = unit;
            m_tensorDataPool.DoubleTensorDataMap[m_tensorDataPool.Counter++] =
                tensorData;
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            m_unitPool.IntUnitMap[m_unitPool.Counter++] = unit;
            m_tensorDataPool.IntTensorDataMap[m_tensorDataPool.Counter++] =
                tensorData;
        }
        else
        {
            static_assert(false, "Unsupported data type");
        }
    }

    for (const auto& [name, tensor] : unit->InternalMap)
    {
        auto* tensorData = Util::TensorData<T>::CreateTensorData(
            tensor.GetShape(), tensor.GetDevice(), false, m_batchSize);

        tensor.RegisterTensorData(tensorData);
        unit->BatchSize = m_batchSize;

        if constexpr (std::is_same_v<T, float>)
        {
            m_unitPool.FloatUnitMap[m_unitPool.Counter++] = unit;
            m_tensorDataPool.FloatTensorDataMap[m_tensorDataPool.Counter++] =
                tensorData;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            m_unitPool.DoubleUnitMap[m_unitPool.Counter++] = unit;
            m_tensorDataPool.DoubleTensorDataMap[m_tensorDataPool.Counter++] =
                tensorData;
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            m_unitPool.IntUnitMap[m_unitPool.Counter++] = unit;
            m_tensorDataPool.IntTensorDataMap[m_tensorDataPool.Counter++] =
                tensorData;
        }
        else
        {
            static_assert(false, "Unsupported data type");
        }
    }
}

template <typename T>
void Model::AutoGrad(Tensor<T> tensor)
{
    while (tensor.PeekTrajectory() >= 0)
    {
        auto key = tensor.PopTrajectory();
        auto* unit = m_unitPool.GetUnit(key);
        auto optionalTensorVector = unit->InvokeBackwardAsyncTensor(tensor);
        if (optionalTensorVector)
        {
            for (auto& outputTensor : optionalTensorVector)
            {
                AutoGrad(outputTensor);
            }
        }
    }
}

}

#endif
