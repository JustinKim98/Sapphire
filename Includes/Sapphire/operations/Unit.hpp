// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UNIT_DECL_HPP
#define SAPPHIRE_UNIT_DECL_HPP

#include <Sapphire/tensor/TensorData.hpp>
#include <unordered_map>
#include <Sapphire/tensor/TensorDescriptor.hpp>


namespace Sapphire
{
class Unit
{
public:
    Unit() = default;
    virtual ~Unit() = default;

    Unit(const Unit& unit) = default;
    Unit(Unit&& unit) noexcept = default;
    Unit& operator=(const Unit& unit) = default;
    Unit& operator=(Unit&& unit) noexcept = default;

protected:
    virtual bool m_checkArguments(
        std::vector<TensorUtil::TensorDescriptor> arguments) = 0;
    std::unordered_map<std::string, TensorUtil::TensorData> m_trainableDataMap;
    std::unordered_map<std::string, TensorUtil::TensorData> m_mutableDataMap;
};

//! UnitDataWrapper
//! Wraps required temporary data of the unit
class UnitDataWrapper
{
public:
    UnitDataWrapper() = default;
    virtual ~UnitDataWrapper() = default;

    UnitDataWrapper(const UnitDataWrapper& unit) = default;
    UnitDataWrapper(UnitDataWrapper&& unitDataWrapper) noexcept = default;
    UnitDataWrapper& operator=(const UnitDataWrapper& unit) = default;
    UnitDataWrapper& operator=(UnitDataWrapper&& unitDataWrapper) noexcept
    = default;

    std::unordered_map<std::string, TensorUtil::TensorData> TensorDataMap;
    std::unordered_map<std::string, std::string> StringLiterals;
    std::unordered_map<std::string, float> ScalarLiterals;
    std::unordered_map<std::string, int> IntegerLiterals;

    std::string Name;
};
} // namespace Sapphire

#endif
