// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_RESOURCEMANAGER_HPP
#define MOTUTAPU_RESOURCEMANAGER_HPP

#include <Motutapu/util/ResourceManagerDecl.hpp>
#include <type_traits>
#include <mutex>

namespace Motutapu
{
template <typename T>
Util::TensorData<T>* TensorPool::GetTensorDataPtr(int tensorId)
{
    static_assert(
        std::disjunction_v<std::is_same<T, float>,
                           std::is_same<T, int>,
                           std::is_same<T, double>>, "Unsupported type");

    auto lock = std::shared_lock(m_mtx);
    if constexpr (std::is_same<T, float>::value)
    {
        return m_tensorDataMapFloat.at(tensorId);
    }
    if constexpr (std::is_same<T, int>::value)
    {
        return m_tensorDataMapInt.at(tensorId);
    }
    if constexpr (std::is_same<T, double>::value)
    {
        return m_tensorDataMapDouble.at(tensorId);
    }

    throw std::runtime_error("Unsupported type");
}

template <typename T>
void TensorPool::InsertTensorData(Util::TensorData<T>* tensorData, int tensorId)
{
    static_assert(
        std::disjunction_v<std::is_same<T, float>, std::is_same<T, int>,
                           std::is_same<T, double>>,
        "Unsupported type");

    auto lock = std::unique_lock(m_mtx);

    if constexpr (std::is_same<T, float>::value)
    {
        m_tensorDataMapFloat.insert_or_assign(tensorId, tensorData);
    }
    if constexpr (std::is_same<T, int>::value)
    {
        m_tensorDataMapInt.insert_or_assign(tensorId, tensorData);
    }
    if constexpr (std::is_same<T, double>::value)
    {
        m_tensorDataMapDouble.insert_or_assign(tensorId, tensorData);
    }
}

template <typename T>
Unit<T>& UnitPool::GetUnit(int unitId)
{
    static_assert(
        std::disjunction_v<std::is_same<T, float>, std::is_same<T, int>,
                           std::is_same<T, double>>,
        "Unsupported type");

    auto lock = std::unique_lock(m_mtx);

    if constexpr (std::is_same<T, float>::value)
    {
        return m_floatUnitMap.at(unitId);
    }
    if constexpr (std::is_same<T, int>::value)
    {
        return m_intUnitMap.at(unitId);
    }
    if constexpr (std::is_same<T, double>::value)
    {
        return m_doubleUnitMap.at(unitId);
    }

    throw std::runtime_error("Unsupported type");
}

template <typename T>
void UnitPool::InsertUnit(Unit<T>& unit, int unitId)
{
    static_assert(
        std::disjunction_v<std::is_same<T, float>, std::is_same<T, int>,
                           std::is_same<T, double>>,
        "Unsupported type");

    auto lock = std::unique_lock(m_mtx);

    if constexpr (std::is_same<T, float>::value)
    {
        m_floatUnitMap.insert_or_assign(unitId, unit);
    }
    if constexpr (std::is_same<T, int>::value)
    {
        m_intUnitMap.insert_or_assign(unitId, unit);
    }
    if constexpr (std::is_same<T, double>::value)
    {
        m_doubleUnitMap.insert_or_assign(unitId, unit);
    }
}
}

#endif
