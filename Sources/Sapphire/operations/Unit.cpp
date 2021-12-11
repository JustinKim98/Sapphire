// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Unit.hpp>

namespace Sapphire
{
void Unit::ToHost()
{
    for (auto& [_, tensor] : m_trainableTensorMap)
    {
        tensor.ToHost();
    }
}

void Unit::ToCuda()
{
    for (auto& [_, tensor] : m_trainableTensorMap)
        tensor.ToCuda();
}
}
