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
    virtual ~Optimizer() = default;
    Optimizer(const Optimizer& optimizer) = default;
    Optimizer(Optimizer&& optimizer) noexcept = default;
    Optimizer& operator=(const Optimizer& optimizer) = default;
    Optimizer& operator=(Optimizer&& optimizer) noexcept = default;

    virtual void operator()(TensorData& z, const TensorData& dz, std::string name)
    {
        throw std::runtime_error(
            "Optimizer::Optimizer::operator() - Default operator should not be "
            "called");
    }
};
}

#endif
