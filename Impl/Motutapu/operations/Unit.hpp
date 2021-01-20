// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UNIT_HPP
#define MOTUTAPU_UNIT_HPP

#include <Motutapu/operations/UnitDecl.hpp>

namespace Motutapu
{
template <typename T>
bool Unit<T>::m_checkBackwardReady()
{
    size_t matches = 0;
    for (auto& [key, inputTensor] : OutputTensorMap)
    {
        for (auto& tensor : BackwardInputTensorPool)
        {
            if (tensor.GetTensorDataKey() == inputTensor.GetTensorDataKey())
                matches += 1;
        }
    }

    if (matches == OutputTensorMap.size())
        return true;
    return false;
}
}

#endif
