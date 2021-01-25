// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UNIT_DECL_HPP
#define MOTUTAPU_UNIT_DECL_HPP

#include <unordered_map>
#include <optional>
#include <Motutapu/tensor/TensorDecl.hpp>

namespace Motutapu
{
template <typename T>
class UnitDataWrapper
{
public:
    UnitDataWrapper() = default;
    virtual ~UnitDataWrapper() = default;

    UnitDataWrapper(const UnitDataWrapper<T>& unit) = default;
    UnitDataWrapper<T>& operator=(const UnitDataWrapper& unit) = default;

    std::unordered_map<std::string, Util::TensorData<T>> TensorDataMap;

    std::unordered_map<std::string, std::string> StringLiterals;
    std::unordered_map<std::string, T> ScalarLiterals;
    std::unordered_map<std::string, int> IntegerLiterals;

    std::string Name;
    Device HostDevice;
    int Key;

};
}


#endif
