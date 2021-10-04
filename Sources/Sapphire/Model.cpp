// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>

namespace Sapphire
{
Model::Model(std::string name)
    : m_name(std::move(name))
{
}

int Model::RegisterTensorDescriptor(const Shape& shape, Type type,
                                    const CudaDevice& device)
{
    const int tensorDescKey = m_tensorDescriptorPool.Counter++;
    TensorUtil::TensorDescriptor tensorDesc(shape, type, device, tensorDescKey);

    m_tensorDescriptorPool.TensorDescMap[tensorDescKey] = std::move(tensorDesc);

    return tensorDescKey;
}

void Model::m_autoGrad(int tensorKey)
{
    if (auto& descriptor = GetDescriptor(tensorKey);
        descriptor.IsBackPropReady())
    {
        descriptor.PopIfOperandHistory();
        if (!descriptor.HasHistory())
        {
            m_removeDescriptor(tensorKey);
            return;
        }

        const auto& [wrapper, location] =
            descriptor.GetBackPropWrapperFromLastHistory();
        const auto outputGradientKeyVector = wrapper->
            GetGradientOutputDescriptorKeys();

        auto data = descriptor.GetBackwardData();

        //! Checks if wrapper is ready to backprop. If it does, performs backprop
        //! Update the operands if successes
        const bool invoked = wrapper->InvokeBackPropIfReady(location);
        descriptor.PopOutputHistory(); //! Pop output history
        if (!descriptor.HasHistory())
            m_removeDescriptor(tensorKey);

        if (invoked)
        {
            for (auto& descKey : outputGradientKeyVector)
                GetDescriptor(descKey).RemoveOperand(tensorKey);

            for (auto& tensorData : outputGradientKeyVector)
                m_autoGrad(tensorData);
        }
    }
}

void Model::m_removeDescriptor(int descKey)
{
    m_tensorDescriptorPool.TensorDescMap.erase(descKey);
}

TensorUtil::TensorDescriptor& Model::GetDescriptor(int descKey)
{
    return m_tensorDescriptorPool.TensorDescMap.at(descKey);
}

void Model::BackProp(Tensor tensor)
{
    m_autoGrad(tensor.TensorDescriptorKey());
}

void Model::Clear()
{
    m_tensorDescriptorPool.TensorDescMap.clear();
}

void Model::InitGradient()
{
    for (auto& [id, tensorDesc] : m_tensorDescriptorPool.TensorDescMap)
        tensorDesc.InitGradient();
}

std::string ModelManager::m_currentModel;

std::unordered_map<std::string, Model> ModelManager::m_modelMap;

Model& ModelManager::GetModel(const std::string& modelName)
{
    return m_modelMap.at(modelName);
}

Model& ModelManager::GetCurrentModel()
{
    return m_modelMap.at(m_currentModel);
}

void ModelManager::SetCurrentModel(const std::string& modelName)
{
    if (m_modelMap.find(modelName) == m_modelMap.end())
        throw std::invalid_argument(
            "ModelManager::SetCurrentModel - Given model name is not "
            "registered");
    m_currentModel = modelName;
}

void ModelManager::AddModel(const std::string& modelName)
{
    m_modelMap.emplace(modelName, Model(modelName));
}
} // namespace Sapphire
