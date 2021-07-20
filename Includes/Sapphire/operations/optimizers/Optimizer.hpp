// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_OPTIMIZER_OPTIMIZER_HPP
#define SAPPHIRE_OPTIMIZER_OPTIMIZER_HPP
#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::Optimizer
{
using namespace TensorUtil;

class Optimizer
{
public:
    Optimizer() = default;
    Optimizer(const Optimizer& other) = default;
    Optimizer(Optimizer&& other) noexcept = default;
    Optimizer& operator=(const Optimizer& other) = default;
    Optimizer& operator=(Optimizer&& other) noexcept = default;
    virtual ~Optimizer() = default;
    virtual void operator()(TensorData& z, TensorData& dz) = 0;
};
}

#endif
