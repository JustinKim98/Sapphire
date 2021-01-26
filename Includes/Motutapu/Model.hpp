// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_MODEL_HPP
#define MOTUTAPU_MODEL_HPP

#include <Motutapu/operations/Unit.hpp>
#include <Motutapu/tensor/TensorDescriptor.hpp>
#include <Motutapu/tensor/Tensor.hpp>
#include <string>
#include <unordered_map>

namespace Motutapu
{
//! Descriptor storing unit key and its stream
struct UnitKeyDescriptor
{
    int Key;
    int StreamId;
};

class Model
{
public:
    Model(size_t batchSize, std::string name);
    ~Model() = default;

    Model(const Model& model) = delete;
    Model(Model&& model) noexcept = default;
    Model& operator=(const Model& model) = delete;
    Model& operator=(Model&& model) noexcept = default;

    //! Registers module
    int RegisterUnitWrapper(UnitDataWrapper& unitWrapper);

    //! Registers and assigns key to tensorDesc
    int RegisterTensorDescriptor(Util::TensorDescriptor& tensorDesc);

    //! Initializes gradients before performing AutoGrad
    void ZeroGrad();

    //! Automatically calculates gradient
    //! \param tensorKey : tensor key to the descriptor to start back propagation
    void AutoGrad(int tensorKey);

    UnitDataWrapper GetUnitDataWrapper(int key);

    //! Converts tensor into vector in 1 dimensional format
    //! \param tensor : tensor to extract data from
    const std::vector<float>& GetData(Tensor tensor);

    //! Sets data directly to the tensor
    void SetData(const std::vector<float>& data);

    Util::TensorDescriptor& GetDescriptor(int key)
    {
        return m_tensorDescriptorPool.GetDescriptor(key);
    }

private:

    class TensorDescriptorPool
    {
    public:
        Util::TensorDescriptor& GetDescriptor(int key)
        {
            return TensorDescMap[key];
        }

        std::unordered_map<int, Util::TensorDescriptor> TensorDescMap;


        int Counter = 0;
    };

    class UnitPool
    {
    public:
        UnitDataWrapper GetUnitDataWrapper(int key)
        {
            return UnitWrapperMap[key];
        }

        std::unordered_map<int, UnitDataWrapper> UnitWrapperMap;


        int Counter = 0;
    };


    //! Order of the operation
    std::list<int> m_operationOrder;
    TensorDescriptorPool m_tensorDescriptorPool;
    UnitPool m_unitPool;
    size_t m_batchSize;
    std::string m_name;
};

//! Singleton class for model management
class ModelManager
{
public:
    static Model& GetModel(const std::string& name);

    static Model& GetCurrentModel();

    static Model& SetModel(const std::string& name);

    static void AddModel(const std::string& name);

private:
    static std::string currentModel;
    static std::unordered_map<std::string, Model> m_modelMap;
};
}

#endif
