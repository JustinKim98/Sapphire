// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_MODEL_HPP
#define MOTUTAPU_MODEL_HPP

#include <Motutapu/operations/Unit.hpp>
#include <Motutapu/tensor/Tensor.hpp>
#include <Motutapu/tensor/TensorDescriptor.hpp>
#include <string>
#include <unordered_map>

namespace Motutapu
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
    int RegisterUnitDataWrapper(UnitDataWrapper& unitDataWrapper);

    //! Creates and registers tensor descriptor
    //! Assigns new key to the given tensorDesc
    int RegisterTensorDescriptor(const Shape& shape, Type type,
                                 const Device& device, unsigned int batchSize);

    //! Initializes gradients before training every epoch
    void ZeroGrad();

    //! Returns unitDataWrapper with given key
    UnitDataWrapper GetUnitDataWrapper(int key) const;

    //! Converts tensor into vector in 1 dimensional format
    //! \param tensor : tensor to extract data from
    [[nodiscard]] const std::vector<float>& GetData(Tensor tensor);

    //! Sets data directly to the tensor
    void SetData(const std::vector<float>& data);

    //! Returns descriptor from the key
    //! \param key : key of the descriptor
    //! \return : tensor descriptor of given key
    TensorUtil::TensorDescriptor& GetDescriptor(int key);

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
    static Model& GetModel(const std::string& name);

    static Model& GetCurrentModel();

    static void SetCurrentModel(const std::string& name);

    static void AddModel(const std::string& name);

 private:
    static std::string m_currentModel;
    static std::unordered_map<std::string, Model> m_modelMap;
};
}  // namespace Motutapu

#endif
