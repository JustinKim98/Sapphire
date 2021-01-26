// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Model.hpp>

namespace Motutapu
{
Model::Model(size_t batchSize, std::string name)
    : m_batchSize(batchSize),
      m_name(std::move(name))
{
}

int Model::RegisterUnitWrapper(UnitDataWrapper& unitWrapper)
{
    const int unitKey = m_unitPool.Counter;
    unitWrapper.Key = unitKey;

    m_unitPool.UnitWrapperMap[unitKey] = unitWrapper;

    return unitKey;
}

int Model::RegisterTensorDescriptor(Util::TensorDescriptor& tensorDesc)
{
    const int tensorDescKey = m_tensorDescriptorPool.Counter;
    tensorDesc.Key = tensorDescKey;

    m_tensorDescriptorPool.TensorDescMap[tensorDescKey] = std::move(tensorDesc);

    return tensorDescKey;
}

void Model::AutoGrad(int tensorKey)
{
    auto& descriptor = GetDescriptor(tensorKey);

    if (descriptor.IsBackPropReady())
    {
        const auto& wrapper = descriptor.GetBackPropWrapper();
        const auto tensorKeys = wrapper->GetOutputTensorKeys();

        std::vector<Util::TensorData> outputTensorDataVector(tensorKeys.size());

        for (size_t i = 0; i < tensorKeys.size(); ++i)
        {
            outputTensorDataVector.at(i) =
                GetDescriptor(tensorKeys.at(i)).BackwardData;
        }

        wrapper->Backward(outputTensorDataVector, descriptor.BackwardData);

        for (const auto key : tensorKeys)
        {
            GetDescriptor(key).RemoveGradientInputKey(tensorKey);
        }

        descriptor.PopHistory();  //! Pop output history
    }
}
}
