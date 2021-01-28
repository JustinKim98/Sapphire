// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Model.hpp>

namespace Motutapu
{
Model::Model(std::string name) : m_name(std::move(name))
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
        descriptor.PopIfOperandHistory();
        const auto& wrapper = descriptor.GetBackPropWrapper();
        const auto outputTensorKeys = wrapper->GetOutputTensorKeys();

        std::vector<Util::TensorData> outputTensorDataVector(
            outputTensorKeys.size());

        for (size_t i = 0; i < outputTensorKeys.size(); ++i)
        {
            outputTensorDataVector.at(i) =
                GetDescriptor(outputTensorKeys.at(i)).BackwardData;
        }

        if (wrapper->IsInplace())
            outputTensorDataVector.emplace_back(descriptor.BackwardData);

        wrapper->Backward(outputTensorDataVector, descriptor.BackwardData);
        descriptor.PopHistory();  //! Pop output history

        if (wrapper->IsInplace())
            descriptor.RemoveGradientInputKey(tensorKey);

        for (const auto key : outputTensorKeys)
        {
            GetDescriptor(key).RemoveGradientInputKey(tensorKey);
            AutoGrad(key);
        }
    }
}

UnitDataWrapper Model::GetUnitDataWrapper(int key)
{
    return m_unitPool.UnitWrapperMap.at(key);
}

Util::TensorDescriptor& Model::GetDescriptor(int key)
{
    return m_tensorDescriptorPool.TensorDescMap.at(key);
}

Model& ModelManager::GetModel(const std::string& name)
{
    return m_modelMap.at(name);
}

Model& ModelManager::GetCurrentModel()
{
    return m_modelMap.at(currentModel);
}

void ModelManager::SetCurrentModel(const std::string& name)
{
    currentModel = name;
}

void ModelManager::AddModel(const std::string& name)
{
    m_modelMap[name] = Model(name);
}
}  // namespace Motutapu
