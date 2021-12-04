// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Backward/LinearBackward.hpp>
#include <Sapphire/compute/Initialize.hpp>

namespace Sapphire::BackProp
{
LinearBackProp::LinearBackProp(std::string name, TensorUtil::TensorData dx,
                               TensorUtil::TensorData dy,
                               TensorUtil::TensorData weight,
                               TensorUtil::TensorData bias,
                               TensorUtil::TensorData x,
                               int batchSize)
    : BackPropWrapper(std::move(name), { std::move(dx) }, { std::move(dy) },
                      { std::move(weight), std::move(bias) },
                      { std::move(x) },
                      {}),
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
    TensorUtil::TensorData weightTranspose(
        weight.GetShape().GetTranspose(), weight.GetType(), weight.GetDevice());
    weightTranspose.SetMode(weight.Mode());

    const auto shape0 = dx.GetShape();
    const auto shape1 = weightTranspose.GetShape();
    const auto shape3 = dy.GetShape();

    Compute::Transpose(weightTranspose, weight);
    Compute::Gemm(dx, dy, weightTranspose);
}

void LinearBackProp::m_updateWeight(TensorUtil::TensorData& weight) const
{
    const TensorUtil::TensorData& dy = m_dyVector[dyIdx];
    const TensorUtil::TensorData& x = m_constants[xIdx];
    TensorUtil::TensorData xTranspose(x.GetShape().GetTranspose(),
                                      x.GetType(),
                                      x.GetDevice());
    TensorUtil::TensorData dw(weight.GetShape(),
                              weight.GetType(), weight.GetDevice());

    xTranspose.SetMode(x.Mode());
    dw.SetMode(weight.Mode());

    const auto shape0 = xTranspose.GetShape();
    const auto shape1 = dw.GetShape();
    const auto shape2 = dy.GetShape();

    Compute::Transpose(xTranspose, x);
    Compute::Initialize::Zeros(dw);
    Compute::Gemm(dw, xTranspose, dy);
    //Compute::Scale(dw, dw, 1.0f / static_cast<float>(m_batchSize));

    ModelManager::CurModel().GetOptimizer()->operator()(weight, dw, m_name);
}

void LinearBackProp::m_updateBias(TensorUtil::TensorData& bias) const
{
    const TensorUtil::TensorData& dy = m_dyVector[dyIdx];
    TensorUtil::TensorData transposedOnes(Shape({ 1, m_batchSize }),
                                          dy.GetType(), dy.GetDevice());
    TensorUtil::TensorData dB(bias.GetShape(), bias.GetType(),
                              bias.GetDevice());

    dB.SetMode(bias.Mode());
    transposedOnes.SetMode(bias.Mode());

    const auto shape0 = transposedOnes.GetShape();
    const auto shape2 = dB.GetShape();
    const auto shape3 = dy.GetShape();

    Compute::Initialize::Ones(transposedOnes);
    Compute::Initialize::Zeros(dB);
    Compute::Gemm(dB, transposedOnes, dy);

    Compute::Scale(dB, dB, 1.0f / static_cast<float>(m_batchSize));
    ModelManager::CurModel().GetOptimizer()->operator()(bias, dB, m_name);
}
} // namespace Sapphire::BackProp
