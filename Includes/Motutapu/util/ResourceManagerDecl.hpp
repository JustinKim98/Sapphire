// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef MOTUTAPU_RESOURCEMANAGER_DECL_HPP
#define MOTUTAPU_RESOURCEMANAGER_DECL_HPP

#include <functional>
#include <Motutapu/util/TensorDataDecl.hpp>
#include <Motutapu/util/ConcurrentQueue.hpp>
#include <string>
#include <unordered_map>
#include <shared_mutex>

#include <Motutapu/tensor/TensorDecl.hpp>

namespace Motutapu
{
template <typename T>
class Unit
{
public:
    std::unordered_map<std::string, int> OutputMap;
    std::unordered_map<std::string, int> InputMap;

    std::unordered_map<std::string, int> InternalVariableTensor;
    std::unordered_map<std::string, int> InternalConstantTensor;

    std::unordered_map<std::string, std::string> StringLiterals;
    std::unordered_map<std::string, T> ScalarLiterals;
    std::unordered_map<std::string, int> IntegerLiterals;

    std::function<std::vector<Tensor<T>>(std::vector<Tensor<T>>&, Unit<T>)>
    BackwardFunction;
    std::function<std::vector<Tensor<T>>(std::vector<Tensor<T>>&, Unit<T>)>
    ForwardFunction;

    std::function<void(Unit<T>)> SaveFunction;
};

class TensorPool
{
public:
    TensorPool() = default;

    template <typename T>
    Util::TensorData<T>* GetTensorDataPtr(int tensorId);

    template <typename T>
    void InsertTensorData(Util::TensorData<T>* tensorData, int tensorId);

private:
    std::unordered_map<int, Util::TensorData<float>*> m_tensorDataMapFloat;
    std::unordered_map<int, Util::TensorData<int>*> m_tensorDataMapInt;
    std::unordered_map<int, Util::TensorData<double>*> m_tensorDataMapDouble;
    std::shared_mutex m_mtx;
};

class UnitPool
{
protected:
    template <typename T>
    void InsertUnit(Unit<T>& unit, int unitId);

    template <typename T>
    Unit<T>& GetUnit(int unitId);

    void Save();

private:
    std::unordered_map<int, Unit<float>>
    m_floatUnitMap;
    std::unordered_map<int, Unit<int>>
    m_intUnitMap;
    std::unordered_map<int, Unit<double>>
    m_doubleUnitMap;

    std::shared_mutex m_mtx;
};


static void AllocateResources();
static void FreeResources();

static std::unique_ptr<TensorPool> GlobalTensorPool;
static std::unique_ptr<UnitPool> GlobalUnitPool;
}

#endif
