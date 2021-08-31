// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/optimizers/SGD.hpp>

namespace Sapphire::Optimizer
{
SGD::SGD(float learningRate)
    : m_learningRate(learningRate)
{
}

SGD::SGD(SGD&& sgd) noexcept
    : Optimizer(sgd),
      m_learningRate(sgd.m_learningRate)
{
}

void SGD::operator()(TensorData& z, const TensorData& dz)
{
    TensorData temp(dz.GetShape(), dz.GetType(), dz.GetCudaDevice());
    Compute::Scale(temp, dz, m_learningRate);
    Compute::Sub(z, z, temp);
}
}
