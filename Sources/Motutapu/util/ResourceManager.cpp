// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/util/ResourceManagerDecl.hpp>


namespace Motutapu
{
void AllocateResources()
{
    GlobalTensorPool = std::make_unique<TensorPool>();
    GlobalUnitPool = std::make_unique<UnitPool>();
}

void FreeResources()
{
    GlobalTensorPool.reset();
    GlobalUnitPool.reset();
}
}
