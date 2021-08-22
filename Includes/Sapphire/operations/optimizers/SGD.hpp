// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_OPTIMIZER_SGD_HPP
#define SAPPHIRE_OPTIMIZER_SGD_HPP

#include <Sapphire/operations/optimizers/Optimizer.hpp>
#include <Sapphire/compute/BasicOps.hpp>

namespace Sapphire::Optimizer
{
class SGD final : public Optimizer
{
public:
    SGD(float learningRate);

    explicit SGD(const SGD& sgd) = default;
    explicit SGD(SGD&& sgd) noexcept;
    ~SGD() override = default;
    SGD& operator=(const SGD& sgd) = default;
    SGD& operator=(SGD&& sgd) noexcept = default;

    void operator()(TensorData& z, TensorData& dz) override;

private:
    float m_learningRate;
};
}

#endif
