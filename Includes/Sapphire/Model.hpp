// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_MODEL_HPP
#define SAPPHIRE_MODEL_HPP

#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/tensor/TensorDescriptor.hpp>
#include <string>
#include <unordered_map>

namespace Sapphire
{
class Model
{
public:
    explicit Model(std::string name);
    ~Model() = default;

    Model(const Model& model) = delete;
    Model(Model&& model) noexcept = default;
    Model& operator=(const Model& model) = delete;
    Model& operator=(Model&& model) noexcept = default;

    //! Registers unitDataWrapper to the unit
    //! Assigns new key to the given unitDataWrapper
    //! \param unitDataWrapper : unitDataWrapper to register
    //! \return : Assigned key
    int AddUnitDataWrapper(UnitDataWrapper& unitDataWrapper);

    void RemoveUnitDataWrapper(int key);

    //! Creates and registers tensor descriptor
    //! Assigns new key to the given tensorDesc
    int RegisterTensorDescriptor(const Shape& shape, Type type,
                                 const Device& device);

    //! Returns unitDataWrapper with given key
    [[nodiscard]] UnitDataWrapper& GetUnitDataWrapper(int key);

    //! Returns descriptor using the descKey
    //! \param descKey : key of the descriptor
    //! \return : tensor descriptor of given key
    [[nodiscard]] TensorUtil::TensorDescriptor& GetDescriptor(int descKey);

    //! Starts back propagation from the given tensor
    //! \param tensor : tensor to start back propagation
    void BackProp(Tensor tensor);

    //! Clears the model
    void Clear();

private:
    //! Automatically calculates gradient
    //! \param tensorKey : tensor key to the descriptor to start back
    //! propagation
    void m_autoGrad(int tensorKey);

    class TensorDescriptorPool
    {
    public:
        std::unordered_map<int, TensorUtil::TensorDescriptor> TensorDescMap;
        int Counter = 0;
    };

    class UnitPool
    {
    public:
        std::unordered_map<int, UnitDataWrapper> UnitWrapperMap;
        int Counter = 0;
    };

    TensorDescriptorPool m_tensorDescriptorPool;
    UnitPool m_unitPool;
    std::string m_name;
};

//! Singleton class for model management
class ModelManager
{
public:
    //! Gets the model with given modelName
    //! \param modelName : name of the model to get
    static Model& GetModel(const std::string& modelName);

    //! Returns currently active model
    //! \return : current model
    static Model& GetCurrentModel();

    //! Sets current model to the given modelName
    //! \param modelName : name of the model to be set
    static void SetCurrentModel(const std::string& modelName);

    //! Adds a new model to the ModelManager
    //! \param modelName : name of the model
    static void AddModel(const std::string& modelName);

private:
    static std::string m_currentModel;
    static std::unordered_map<std::string, Model> m_modelMap;
};
} // namespace Sapphire

#endif
