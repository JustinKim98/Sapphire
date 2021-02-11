// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_LINEARBACKWARD_HPP
#define MOTUTAPU_LINEARBACKWARD_HPP

#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/compute/Initialize.hpp>
#include <Motutapu/operations/Backward/BackPropWrapper.hpp>

namespace Motutapu::BackProp
{
class LinearBackProp : public BackPropWrapper
{
 public:
    explicit LinearBackProp(TensorUtil::TensorData x, TensorUtil::TensorData dx,
                            TensorUtil::TensorData dy, int unitKey)
        : BackPropWrapper({ std::move(dx) }, { std::move(dy), std::move(x) },
                          unitKey)
    {
        TensorUtil::TensorData& gradA = m_gradientOutputs[0];
        TensorUtil::TensorData& gradIn = m_gradientInputs[0];
        TensorUtil::TensorData& a = m_gradientInputs[1];

        //! Treat x and gradA
        a.TensorShape.Expand(2);
        gradA.TensorShape.Expand(2);
        gradIn.TensorShape.Expand(2);
        a.TensorShape[0] = a.BatchSize;
        gradA.TensorShape[0] = gradA.BatchSize;
        gradIn.TensorShape[0] = gradIn.BatchSize;
        a.BatchSize = 1;
        gradA.BatchSize = 1;
        gradIn.BatchSize = 1;
    }

    bool InvokeBackProp(const TensorUtil::TensorData& input) override
    {
        const auto& model = ModelManager::GetCurrentModel();
        auto unitDataWrapper = model.GetUnitDataWrapper(m_unitKey);
        auto weight = unitDataWrapper.TensorDataMap["weight"];
        auto bias = unitDataWrapper.TensorDataMap["bias"];

        m_backProp(weight);
        m_updateWeight(weight);
        m_updateBias(bias);

        return true;
    }

 private:
    void m_backProp(const TensorUtil::TensorData& weight)
    {
        TensorUtil::TensorData transposedWeight(
            weight.GetShape().GetTranspose(), weight.GetType(),
            weight.GetDevice(), 1);
        TensorUtil::TensorData& gradientA = m_gradientOutputs[0];
        TensorUtil::TensorData& gradientIn = m_gradientInputs[0];

        Compute::Initialize::Zeros(gradientA);
        Compute::Transpose(transposedWeight, weight);
        Compute::Gemm(gradientA, gradientIn, transposedWeight, gradientA);
    }

    void m_updateWeight(TensorUtil::TensorData& weight)
    {
        TensorUtil::TensorData& gradientIn = m_gradientInputs[0];
        TensorUtil::TensorData& A = m_gradientInputs[1];
        TensorUtil::TensorData transposedA(A.GetShape().GetTranspose(),
                                           A.GetType(), A.GetDevice(), 1);

        Compute::Transpose(transposedA, A);
        Compute::Scale(
            transposedA, transposedA,
            -1);  // todo : divide by batch size and scale by learning rate
        Compute::Gemm(weight, transposedA, gradientIn, weight);
    }

    void m_updateBias(TensorUtil::TensorData& bias)
    {
        TensorUtil::TensorData& gradientIn = m_gradientInputs[0];
        TensorUtil::TensorData oneVector(Shape({ gradientIn.Rows() }),
                                         gradientIn.GetType(),
                                         gradientIn.GetDevice(), 1);

        Compute::Initialize::Ones(oneVector);
        Compute::Scale(oneVector, oneVector, -1.0f);
        Compute::Gemm(bias, oneVector, gradientIn, bias);
    }
};

}  // namespace Motutapu::BackProp

#endif  // MOTUTAPU_LINEARBACKWARD_HPP
