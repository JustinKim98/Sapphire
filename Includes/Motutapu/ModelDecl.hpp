// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_MODEL_DECL_HPP
#define MOTUTAPU_MODEL_DECL_HPP

#include <Motutapu/operations/Unit.hpp>
#include <Motutapu/tensor/TensorDescriptor.hpp>
#include <string>
#include <unordered_map>
#include <shared_mutex>

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
    template <typename T>
    int RegisterUnitWrapper(UnitDataWrapper<T>& unitWrapper);

    //! Registers and assigns key to tensorDesc
    template<typename T>
    int RegisterTensorDescriptor(Util::TensorDescriptor<T>& tensorDesc);

    //! Initializes gradients before performing AutoGrad
    void ZeroGrad();

    //! Automatically calculates gradient
    //! \tparam T : template type for the data
    //! \param tensorKey : tensor key to the descriptor to start back propagation
    template <typename T>
    void AutoGrad(int tensorKey);

    template <typename T>
    UnitDataWrapper<T>* GetUnitDataWrapper(int key);

    //! Converts tensor into vector in 1 dimensional format
    //! \tparam T : template type for the data
    //! \param tensor : tensor to extract data from
    template <typename T>
    const std::vector<T>& GetData(Tensor<T> tensor);

    //! Sets data directly to the tensor
    template <typename T>
    void SetData(const std::vector<T>& data);

    template <typename T>
    Util::TensorDescriptor<T>& GetDescriptor(int key)
    {
        return m_tensorDescriptorPool.GetDescriptor<T>(key);
    }

private:

    class TensorDescriptorPool
    {
    public:
        template <typename T>
        Util::TensorDescriptor<T>& GetDescriptor(int key)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return FloatTensorDescMap[key];
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return DoubleTensorDescMap[key];
            }
            else if constexpr (std::is_same_v<T, int>)
            {
                return IntTensorDescMap[key];
            }

            throw std::runtime_error("GetDescriptor - Unsupported type");
        }

        std::unordered_map<int, Util::TensorDescriptor<float>> FloatTensorDescMap;
        std::unordered_map<int, Util::TensorDescriptor<double>> DoubleTensorDescMap;
        std::unordered_map<int, Util::TensorDescriptor<int>> IntTensorDescMap;

        int Counter = 0;
    };

    class UnitPool
    {
    public:
        template <typename T>
        UnitDataWrapper<T> GetUnitDataWrapper(int key)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return FloatUnitMap[key];
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return DoubleUnitMap[key];
            }
            else if constexpr (std::is_same_v<T, int>)
            {
                return IntUnitMap[key];
            }
            return nullptr;
        }

        std::unordered_map<int, UnitDataWrapper<float>> FloatUnitMap;
        std::unordered_map<int, UnitDataWrapper<double>> DoubleUnitMap;
        std::unordered_map<int, UnitDataWrapper<int>> IntUnitMap;

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
