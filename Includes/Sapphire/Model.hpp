// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_MODEL_HPP
#define SAPPHIRE_MODEL_HPP

#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/tensor/TensorDescriptor.hpp>
#include <Sapphire/operations/Backward/BackPropWrapper.hpp>
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

    //! Creates and registers tensor descriptor
    //! Assigns new key to the given tensorDesc
    int RegisterTensorDescriptor(const Shape& shape, Type type,
                                 const CudaDevice& device,
                                 bool preserve = false);

    //! Registers back propagation wrapper
//! \param backPropWrapper :  back propagation wrapper to register
//! \return : key of the back propagation wrapper
    int RegisterBackPropWrapper(BackProp::BackPropWrapper* backPropWrapper);

    //! Returns descriptor using the descKey
    //! \param descKey : key of the descriptor
    //! \return : tensor descriptor of given key
    [[nodiscard]] TensorUtil::TensorDescriptor& GetDescriptor(int descKey);

    //! Starts back propagation from the given tensor
    //! \param tensor : tensor to start back propagation
    void BackProp(Tensor tensor);

    //! Clears the model
    //! including forward and back prop data
    void Clear();

    //! Initializes gradients to zero
    void InitGradient();

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

    void m_removeDescriptor(int descKey);


    TensorDescriptorPool m_tensorDescriptorPool;
    TensorDescriptorPool m_preservedDescriptorPool;
    std::unordered_map<int, BackProp::BackPropWrapper*> m_backPropWrapperPool;
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
    static Model& CurModel();

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
