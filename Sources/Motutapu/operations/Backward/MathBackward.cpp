// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/operations/Backward/MathBackward.hpp>
#include <iostream>

namespace Motutapu::BackProp
{
MulBackProp::MulBackProp(TensorUtil::TensorData a, TensorUtil::TensorData da,
                         TensorUtil::TensorData b, TensorUtil::TensorData db,
                         TensorUtil::TensorData dy)
    : BackPropWrapper({ std::move(da), std::move(db) }, { std::move(dy) })
{
    TensorUtil::TensorData transposedA(a.TensorShape.GetTranspose(),
                                       a.GetType(), a.GetDevice(), a.BatchSize);
    TensorUtil::TensorData transposedB(b.TensorShape.GetTranspose(),
                                       b.GetType(), b.GetDevice(), b.BatchSize);

    m_savedTensorMap["a"] = std::move(a);
    m_savedTensorMap["b"] = std::move(b);
    m_savedTensorMap["transposedA"] = transposedA;
    m_savedTensorMap["transposedB"] = transposedB;
}

bool MulBackProp::InvokeBackProp(const TensorUtil::TensorData& input)
{
    auto& dy = m_gradientInputs[0];
    auto& da = m_gradientOutputs[0];
    auto& db = m_gradientOutputs[1];

    auto& a = m_savedTensorMap["a"];
    auto& b = m_savedTensorMap["b"];
    auto& transposedA = m_savedTensorMap["transposedA"];
    auto& transposedB = m_savedTensorMap["transposedB"];

    Compute::Transpose(transposedA, a);
    Compute::Transpose(transposedB, b);

    Compute::Gemm(da, dy, transposedB, da);
    Compute::Gemm(db, transposedA, dy, db);
    return true;
}

AddBackProp::AddBackProp(TensorUtil::TensorData da, TensorUtil::TensorData db,
                         TensorUtil::TensorData dy)
    : BackPropWrapper({ std::move(da), std::move(db) }, { std::move(dy) })
{
}

bool AddBackProp::InvokeBackProp(const TensorUtil::TensorData& input)
{
    auto& dy = m_gradientInputs[0];
    auto& da = m_gradientOutputs[0];
    auto& db = m_gradientOutputs[1];
    Compute::Add(da, dy, da);
    Compute::Add(db, dy, db);
    return true;
}

}  // namespace Motutapu::BackProp