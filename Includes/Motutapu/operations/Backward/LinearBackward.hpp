// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_LINEARBACKWARD_HPP
#define MOTUTAPU_LINEARBACKWARD_HPP

#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/operations/Backward/BackPropWrapper.hpp>

namespace Motutapu::BackProp
{
class LinearBackProp : public BackPropWrapper
{
 public:
    explicit LinearBackProp(unsigned int gradientOutputTensorKey, int unitKey)
        : BackPropWrapper({ gradientOutputTensorKey }, false, unitKey)
    {
    }

    void Backward(std::vector<TensorUtil::TensorData>& outputs,
                  const TensorUtil::TensorData& input) const override
    {
        auto& model = ModelManager::GetCurrentModel();
        const auto inputShape = input.GetShape();
        auto unitDataWrapper = model.GetUnitDataWrapper(m_unitKey);

        TensorUtil::TensorData weight = unitDataWrapper.TensorDataMap["weight"];
        TensorUtil::TensorData transposedWeight =
            unitDataWrapper.TensorDataMap["TransposedWeight"];

        //! Calculate next gradient
        Compute::Transpose(transposedWeight, input);
        Compute::Mul(outputs[0], input, transposedWeight);
    }
};

//!
class LinearOptimize
{
    void Optimize();
};
}  // namespace Motutapu::BackProp

#endif  // MOTUTAPU_LINEARBACKWARD_HPP
