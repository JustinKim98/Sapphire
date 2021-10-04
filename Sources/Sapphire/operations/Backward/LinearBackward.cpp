// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Backward/LinearBackward.hpp>
#include <Sapphire/compute/Initialize.hpp>

namespace Sapphire::BackProp
{
LinearBackProp::LinearBackProp(TensorUtil::TensorData dx,
                               TensorUtil::TensorData dy,
                               TensorUtil::TensorData weight,
                               TensorUtil::TensorData bias,
                               TensorUtil::TensorData x,
                               Util::SharedPtr<Optimizer::Optimizer> optimizer,
                              int batchSize)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) },
                      { std::move(weight), std::move(bias) },
                      { std::move(x) },
                      {}, std::move(optimizer)),
      m_batchSize(batchSize)
{
}

void LinearBackProp::m_runBackProp()
{
    auto weight = m_trainableData[weightIdx];
    auto bias = m_trainableData[biasIdx];

    m_backProp(weight);
    m_updateWeight(weight);
    m_updateBias(bias);
}

void LinearBackProp::m_backProp(TensorUtil::TensorData& weight)
{
    TensorUtil::TensorData& dx = m_dxVector[dxIdx];
    TensorUtil::TensorData& dy = m_dyVector[dyIdx];

    Compute::Gemm(dx, dy, weight, dx);
}

void LinearBackProp::m_updateWeight(TensorUtil::TensorData& weight) const
{
    const TensorUtil::TensorData& dy = m_dyVector[dyIdx];
    const TensorUtil::TensorData& x = m_constants[xIdx];
    TensorUtil::TensorData xTranspose(x.GetShape().GetTranspose(),
                                      x.GetType(),
                                      x.GetDevice());
    TensorUtil::TensorData dyTranspose(dy.GetShape().GetTranspose(),
                                       dy.GetType(), dy.GetDevice());
    TensorUtil::TensorData dw(weight.GetShape().GetTranspose(),
                              weight.GetType(), weight.GetDevice());

    xTranspose.SetMode(x.Mode());
    dyTranspose.SetMode(dy.Mode());
    dw.SetMode(weight.Mode());

    Compute::Transpose(xTranspose, x);
    Compute::Transpose(dyTranspose, dy);
    Compute::Initialize::Zeros(dw);
    Compute::Gemm(dw, dyTranspose, x, dw);
    Compute::Scale(dw, dw, 1.0f / static_cast<float>(m_batchSize));

    m_optimizer->operator()(weight, dw);
}

void LinearBackProp::m_updateBias(TensorUtil::TensorData& bias) const
{
    const TensorUtil::TensorData& dy = m_dyVector[dyIdx];
    TensorUtil::TensorData ones(Shape({ m_batchSize }),
                                dy.GetType(),
                                dy.GetDevice(), 1);
    TensorUtil::TensorData dB(bias.GetShape(), bias.GetType(),
                              bias.GetDevice(), 1);

    dB.SetMode(bias.Mode());
    ones.SetMode(bias.Mode());

    Compute::Initialize::Ones(ones);
    Compute::Initialize::Zeros(dB);
    Compute::Gemm(dB, ones, dy, dB);
    Compute::Scale(dB, dB, 1.0f / static_cast<float>(m_batchSize));
    m_optimizer->operator()(bias, dB);
}
} // namespace Sapphire::BackProp
