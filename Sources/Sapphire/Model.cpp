// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>

namespace Sapphire
{
Model::Model(std::string name) : m_name(std::move(name))
{
}

int Model::RegisterUnitDataWrapper(UnitDataWrapper& unitDataWrapper)
{
    const int unitKey = m_unitPool.Counter++;
    unitDataWrapper.Key = unitKey;

    m_unitPool.UnitWrapperMap[unitKey] = unitDataWrapper;

    return unitKey;
}

//! Todo : Demarcate between tensor with back propagation
int Model::RegisterTensorDescriptor(const Shape& shape, Type type,
                                    const Device& device,
                                    unsigned int batchSize,
                                    bool createBackwardData)
{
    const int tensorDescKey = m_tensorDescriptorPool.Counter++;
    TensorUtil::TensorDescriptor tensorDesc(shape, type, device, batchSize,
                                            tensorDescKey);
    if (createBackwardData)
    {
        tensorDesc.BackwardData = TensorUtil::TensorData(
            shape, type, device, batchSize, tensorDescKey);
    }

    m_tensorDescriptorPool.TensorDescMap[tensorDescKey] = std::move(tensorDesc);

    return tensorDescKey;
}

void Model::m_autoGrad(int tensorKey)
{
    auto& descriptor = GetDescriptor(tensorKey);
    if (descriptor.IsBackPropReady())
    {
        descriptor.PopIfOperandHistory();
        const auto& wrapper = descriptor.GetBackPropWrapper();
        const auto outputTensorDataVector = wrapper->GetOutputTensorKeys();

        bool isInvoked = wrapper->InvokeBackProp(descriptor.BackwardData);
        descriptor.PopHistory();  //! Pop output history

        if (isInvoked)
            for (auto& tensorData : outputTensorDataVector)
            {
                GetDescriptor(tensorData.GetParentDescKey())
                    .RemoveGradientInputKey(tensorKey);
                m_autoGrad(tensorData.GetParentDescKey());
            }
    }
}

UnitDataWrapper Model::GetUnitDataWrapper(int key) const
{
    return m_unitPool.UnitWrapperMap.at(key);
}

TensorUtil::TensorDescriptor& Model::GetDescriptor(int key)
{
    return m_tensorDescriptorPool.TensorDescMap.at(key);
}

std::string ModelManager::m_currentModel;

std::unordered_map<std::string, Model> ModelManager::m_modelMap;

Model& ModelManager::GetModel(const std::string& name)
{
    return m_modelMap.at(name);
}

Model& ModelManager::GetCurrentModel()
{
    return m_modelMap.at(m_currentModel);
}

void ModelManager::SetCurrentModel(const std::string& name)
{
    m_currentModel = name;
}

void ModelManager::AddModel(const std::string& name)
{
    m_modelMap.emplace(name, Model(name));
}
}  // namespace Sapphire
