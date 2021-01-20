// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_MODEL_DECL_HPP
#define MOTUTAPU_MODEL_DECL_HPP

#include <Motutapu/tensor/TensorDataDecl.hpp>
#include <Motutapu/util/ConcurrentQueue.hpp>
#include <Motutapu/operations/UnitDecl.hpp>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include <Motutapu/tensor/TensorDecl.hpp>

namespace Motutapu
{
class Model
{
public:
    Model(size_t batchSize, std::string name);
    ~Model();


    Model(const Model& model) = delete;
    Model(Model&& model) noexcept = default;
    Model& operator=(const Model& model) = delete;
    Model& operator=(Model&& model) noexcept = default;

    //! Registers module
    template <typename T>
    void Register(Unit<T>* unit);

    //! Automatically calculates gradient
    //! \tparam T : template type for the data
    //! \param tensor : tensor to extract 
    template <typename T>
    void AutoGrad(Tensor<T> tensor);

    //! Converts tensor into vector in 1 dimensional format
    //! \tparam T : template type for the data
    //! \param tensor : tensor to extract data from
    template <typename T>
    const std::vector<T>& GetData(Tensor<T> tensor);

    //! Sets data directly to the tensor
    template <typename T>
    void SetData(const std::vector<T>& data);

private:

    class TensorDataPool
    {
    public:
        std::unordered_map<int, Util::TensorData<float>*> FloatTensorDataMap;
        std::unordered_map<int, Util::TensorData<double>*> DoubleTensorDataMap;
        std::unordered_map<int, Util::TensorData<int>*> IntTensorDataMap;

        int Counter = 0;
    };

    class UnitPool
    {
    public:
        template <typename T>
        Unit<T>* GetUnit(int key)
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
            else
            {
                static_assert(false, "Unsupported data type");
            }

            return nullptr;
        }

        std::unordered_map<int, Unit<float>*> FloatUnitMap;
        std::unordered_map<int, Unit<double>*> DoubleUnitMap;
        std::unordered_map<int, Unit<int>*> IntUnitMap;

        int Counter = 0;
    };

    TensorDataPool m_tensorDataPool;
    UnitPool m_unitPool;
    size_t m_batchSize;
    std::string m_name;
};
}

#endif
