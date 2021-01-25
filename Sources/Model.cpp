// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Model.hpp>

namespace Motutapu
{
Model::Model(size_t batchSize, std::string name)
    : m_batchSize(batchSize),
      m_name(std::move(name))
{
}
}
