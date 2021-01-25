// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_MODEL_HPP
#define MOTUTAPU_MODEL_HPP

#include <Motutapu/ModelDecl.hpp>
#include <type_traits>

namespace Motutapu
{
template <typename T>
int Model::RegisterUnitWrapper(UnitDataWrapper<T>& unitWrapper)
{
    const int unitKey = m_unitPool.Counter;
    unitWrapper->Key = unitKey;

    if constexpr (std::is_same_v<T, float>)
    {
        m_unitPool.FloatUnitMap[unitKey] = unitWrapper;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        m_unitPool.DoubleUnitMap[unitKey] = unitWrapper;
    }
    else if constexpr (std::is_same_v<T, int>)
    {
        m_unitPool.IntUnitMap[unitKey] = unitWrapper;
    }

    return unitKey;
}

template <typename T>
int Model::RegisterTensorDescriptor(Util::TensorDescriptor<T>& tensorDesc)
{
    const int tensorDescKey = m_tensorDescriptorPool.Counter;
    tensorDesc->Key = tensorDescKey;

    if constexpr (std::is_same_v<T, float>)
    {
        m_tensorDescriptorPool.FloatTensorDescMap[tensorDescKey] = tensorDesc;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        m_tensorDescriptorPool.DoubleTensorDescMap[tensorDescKey] = tensorDesc;
    }
    else if constexpr (std::is_same_v<T, int>)
    {
        m_tensorDescriptorPool.IntTensorDescMap[tensorDescKey] = tensorDesc;
    }

    return tensorDescKey;
}

template <typename T>
void Model::AutoGrad(int tensorKey)
{
    auto& descriptor = GetDescriptor<T>(tensorKey);

    if (descriptor.IsBackPropReady())
    {
        const auto& wrapper = descriptor.GetBackPropWrapper();
        const auto tensorKeys = wrapper->GetOutputTensorKeys();

        std::vector<Util::TensorData<T>> outputTensorDataVector(tensorKeys);

        for (int i = 0; i < tensorKeys.size(); ++i)
        {
            outputTensorDataVector.at(i) =
                GetDescriptor<T>(tensorKeys.at(i)).BackwardData;
        }

        wrapper->Backward(outputTensorDataVector, descriptor);

        for (const auto key : tensorKeys)
        {
            GetDescriptor<T>(key).RemoveGradientInputKey(tensorKey);
        }

        descriptor.PopHistory(); //! Pop output history
    }
}
}

#endif
