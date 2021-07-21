// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Backward/LinearBackward.hpp>
#include <Sapphire/compute/Initialize.hpp>

namespace Sapphire::BackProp
{
LinearBackProp::LinearBackProp(TensorUtil::TensorData& dx,
                               TensorUtil::TensorData& weight,
                               TensorUtil::TensorData& bias,
                               const TensorUtil::TensorData& dy,
                               const TensorUtil::TensorData& x,
                               std::weak_ptr<Optimizer::Optimizer> optimizer,
                               unsigned int batchSize,
                               int unitKey)
    : BackPropWrapper({ dx }, { dy }, { weight, bias }, std::move(optimizer),
                      unitKey),
      m_batchSize(batchSize)
{
    m_savedTensorMap["x"] = x;
}

bool LinearBackProp::InvokeBackProp(const TensorUtil::TensorData& dy)
{
    auto weight = m_trainableData[weightIdx];
    auto bias = m_trainableData[biasIdx];
    TensorUtil::TensorData::DeepCopy(m_gradientInputs[0], dy);

    m_backProp(weight);
    m_updateWeight(weight);
    m_updateBias(bias);

    return true;
}

void LinearBackProp::m_backProp(const TensorUtil::TensorData& weight)
{
    TensorUtil::TensorData& dx = m_gradientOutputs[dxIdx];
    TensorUtil::TensorData& dy = m_gradientInputs[dyIdx];

    Compute::Initialize::Zeros(dx);
    Compute::Gemm(dx, dy, weight, dx);
}

void LinearBackProp::m_updateWeight(TensorUtil::TensorData& weight)
{
    TensorUtil::TensorData& dy = m_gradientInputs[dyIdx];
    const TensorUtil::TensorData& x = m_savedTensorMap["x"];
    TensorUtil::TensorData transposedX(x.GetShape().GetTranspose(), x.GetType(),
                                       x.GetDevice(), 1);
    TensorUtil::TensorData transposedDy(dy.GetShape().GetTranspose(),
                                        dy.GetType(), dy.GetDevice(), 1);
    TensorUtil::TensorData dw(weight.GetShape().GetTranspose(),
                              weight.GetType(), weight.GetDevice(), 1);

    Compute::Transpose(transposedX, x);
    Compute::Transpose(transposedDy, dy);
    Compute::Initialize::Zeros(dw);
    Compute::Gemm(dw, transposedX, dy, dw);
    Compute::Scale(dw, dw, 1.0f / static_cast<float>(m_batchSize));
    m_optimizer.lock()->operator()(weight, dw);
}

void LinearBackProp::m_updateBias(TensorUtil::TensorData& bias)
{
    TensorUtil::TensorData& dy = m_gradientInputs[dyIdx];
    const TensorUtil::TensorData ones(Shape({ m_batchSize }),
                                      dy.GetType(),
                                      dy.GetDevice(), 1);
    TensorUtil::TensorData dB(bias.GetShape(), bias.GetType(),
                              bias.GetDevice(), 1);

    Compute::Initialize::Ones(ones);
    Compute::Initialize::Zeros(dB);
    Compute::Gemm(dB, ones, dy, dB);
    Compute::Scale(dB, dB, 1.0f / static_cast<float>(m_batchSize));
    m_optimizer.lock()->operator()(bias, dB);
}
} // namespace Sapphire::BackProp
