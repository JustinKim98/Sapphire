// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/MathBackward.hpp>
#include <iostream>

namespace Sapphire::BackProp
{
MulBackProp::MulBackProp(const TensorUtil::TensorData& a,
                         TensorUtil::TensorData da,
                         const TensorUtil::TensorData& b,
                         TensorUtil::TensorData db, TensorUtil::TensorData dy)
    : BackPropWrapper({ std::move(da), std::move(db) }, { std::move(dy) },
                      { a, b },
                      { TensorUtil::TensorData(a.GetShape().GetTranspose(),
                                               a.GetType(),
                                               a.GetDevice()),
                        TensorUtil::TensorData(b.GetShape().GetTranspose(),
                                               b.GetType(),
                                               b.GetDevice()) })
{
    auto& aData = m_constants[0];
    auto& bData = m_constants[1];
    auto& transposedA = m_mutables[0];
    auto& transposedB = m_mutables[1];
    transposedA.SetMode(aData.Mode());
    transposedB.SetMode(bData.Mode());
}

void MulBackProp::m_runBackProp()
{
    auto& dy = m_dyVector[0];
    auto& da = m_dxVector[0];
    auto& db = m_dxVector[1];

    auto& a = m_constants[0];
    auto& b = m_constants[1];
    auto& transposedA = m_mutables[0];
    auto& transposedB = m_mutables[1];

    Compute::Transpose(transposedA, a);
    Compute::Transpose(transposedB, b);

    dy.ToHost();
    b.ToHost();
    transposedB.ToHost();

    Compute::Gemm(da, dy, transposedB, da);
    Compute::Gemm(db, transposedA, dy, db);
}

AddBackProp::AddBackProp(TensorUtil::TensorData da, TensorUtil::TensorData db,
                         TensorUtil::TensorData dy)
    : BackPropWrapper({ std::move(da), std::move(db) }, { std::move(dy) })
{
}

void AddBackProp::m_runBackProp()
{
    const auto& dy = m_dyVector[0];
    auto& da = m_dxVector[0];
    auto& db = m_dxVector[1];
    Compute::Add(da, dy, da);
    Compute::Add(db, dy, db);
}

MeanBackProp::MeanBackProp(TensorUtil::TensorData dx, TensorUtil::TensorData x,
                           TensorUtil::TensorData dy, unsigned int dim)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) }, { std::move(x) },
                      {}),
      m_dim(dim)
{
}

void MeanBackProp::m_runBackProp()
{
    auto& dx = m_dxVector[0];
    const auto& dy = m_dyVector[0];
    const auto& x = m_constants[0];
    Compute::MeanBackward(dx, dy, x, m_dim);
}
} // namespace Sapphire::BackProp
