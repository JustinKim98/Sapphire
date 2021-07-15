// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Backward/LinearBackward.hpp>

#include <Sapphire/compute/Initialize.hpp>

namespace Sapphire::BackProp
{
LinearBackProp::LinearBackProp(const TensorUtil::TensorData& x,
                               TensorUtil::TensorData dx,
                               TensorUtil::TensorData dy, int unitKey)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) }, unitKey),
      m_batchSize(dy.BatchSize)
{
    m_savedTensorMap["x"] = x.CreateCopy();

    TensorUtil::TensorData& dxRef = m_gradientOutputs[0];
    TensorUtil::TensorData& dyRef = m_gradientInputs[0];
    TensorUtil::TensorData& xRef = m_savedTensorMap["x"];

    //! Treat x and dxRef
    xRef.TensorShape.Expand(2);
    dxRef.TensorShape.Expand(2);
    dyRef.TensorShape.Expand(2);
    xRef.TensorShape[0] = xRef.BatchSize;
    dxRef.TensorShape[0] = dxRef.BatchSize;
    dyRef.TensorShape[0] = dyRef.BatchSize;
    xRef.BatchSize = 1;
    dxRef.BatchSize = 1;
    dyRef.BatchSize = 1;
}

bool LinearBackProp::InvokeBackProp(const TensorUtil::TensorData& dy)
{
    auto& model = ModelManager::GetCurrentModel();
    auto unitDataWrapper = model.GetUnitDataWrapper(m_unitKey);
    auto weight = unitDataWrapper.TensorDataMap["weight"];
    auto bias = unitDataWrapper.TensorDataMap["bias"];
    TensorUtil::TensorData::CopyTensorData(m_gradientInputs[0], dy);

    m_backProp(weight);
    m_updateWeight(weight);
    m_updateBias(bias);

    return true;
}

void LinearBackProp::m_backProp(const TensorUtil::TensorData& weight)
{
    TensorUtil::TensorData transposedWeight(weight.GetShape().GetTranspose(),
                                            weight.GetType(),
                                            weight.GetDevice(), 1);
    TensorUtil::TensorData& dx = m_gradientOutputs[0];
    TensorUtil::TensorData& dy = m_gradientInputs[0];

    Compute::Initialize::Zeros(dx);
    Compute::Transpose(transposedWeight, weight);
    Compute::Gemm(dx, dy, transposedWeight, dx);
}

void LinearBackProp::m_updateWeight(TensorUtil::TensorData& weight)
{
    TensorUtil::TensorData& dy = m_gradientInputs[0];
    TensorUtil::TensorData& x = m_savedTensorMap["x"];
    TensorUtil::TensorData transposedA(x.GetShape().GetTranspose(), x.GetType(),
                                       x.GetDevice(), 1);

    Compute::Transpose(transposedA, x);
    Compute::Scale(
        transposedA, transposedA,
        -1 / static_cast<float>(m_batchSize)); // todo : divide by batch size
    // and scale by learning rate
    Compute::Gemm(weight, transposedA, dy, weight);
}

void LinearBackProp::m_updateBias(TensorUtil::TensorData& bias)
{
    TensorUtil::TensorData& gradientIn = m_gradientInputs[0];
    TensorUtil::TensorData oneVector(Shape({ gradientIn.Rows() }),
                                     gradientIn.GetType(),
                                     gradientIn.GetDevice(), 1);

    Compute::Initialize::Ones(oneVector);
    Compute::Scale(oneVector, oneVector, -1.0f);
    Compute::Gemm(bias, oneVector, gradientIn, bias);
}
} // namespace Sapphire::BackProp
