// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/operations/Backward/MathBackward.hpp>
#include <iostream>

namespace Motutapu::BackProp
{
MulBackProp::MulBackProp(TensorUtil::TensorData forwardA,
                         TensorUtil::TensorData backwardA,
                         TensorUtil::TensorData forwardB,
                         TensorUtil::TensorData backwardB,
                         TensorUtil::TensorData backwardOutput)
    : BackPropWrapper({ std::move(backwardA), std::move(backwardB) },
                      { std::move(backwardOutput) })
{
    TensorUtil::TensorData transposedA(forwardA.TensorShape.GetTranspose(),
                                       forwardA.GetType(), forwardA.GetDevice(),
                                       forwardA.BatchSize);
    TensorUtil::TensorData transposedB(forwardB.TensorShape.GetTranspose(),
                                       forwardB.GetType(), forwardB.GetDevice(),
                                       forwardB.BatchSize);

    m_savedTensorMap["forwardA"] = forwardA.CreateCopy();
    m_savedTensorMap["forwardB"] = forwardB.CreateCopy();
    m_savedTensorMap["transposedA"] = transposedA;
    m_savedTensorMap["transposedB"] = transposedB;
}

bool MulBackProp::InvokeBackProp(const TensorUtil::TensorData& input)
{
    auto key = input.GetParentDescKey();
    auto match = [key](const TensorUtil::TensorData& tensorData) {
        return tensorData.GetParentDescKey() == key;
    };

    auto tensorItr =
        std::find_if(m_gradientInputs.begin(), m_gradientInputs.end(), match);

    auto tensorItrReceived =
        std::find_if(m_receivedInputs.begin(), m_receivedInputs.end(), match);

    if (tensorItr != m_gradientInputs.end() &&
        tensorItrReceived != m_receivedInputs.end())
        m_receivedInputs.emplace_back(*tensorItr);

    if (m_receivedInputs.size() == m_gradientInputs.size())
    {
        m_backProp();
        return true;
    }

    return false;
}

void MulBackProp::m_backProp()
{
    auto& backPropA = m_gradientOutputs[0];
    auto& backPropB = m_gradientOutputs[1];
    auto& backPropIn = m_gradientInputs[0];

    auto& forwardOutA = m_savedTensorMap["backPropA"];
    auto& forwardOutB = m_savedTensorMap["forwardB"];
    auto& transposedA = m_savedTensorMap["transposedA"];
    auto& transposedB = m_savedTensorMap["transposedB"];

    Compute::Transpose(transposedA, forwardOutA);
    Compute::Transpose(transposedB, forwardOutB);

    Compute::Gemm(backPropA, backPropIn, transposedB, backPropA);
    Compute::Gemm(backPropB, transposedA, backPropIn, backPropB);
}

}  // namespace Motutapu::BackProp