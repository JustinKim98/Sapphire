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

    void Backward(std::vector<Util::TensorData>& outputs,
                  const Util::TensorData& input) const override
    {
        auto& model = ModelManager::GetCurrentModel();
        const auto inputShape = input.GetShape();
        const auto& device = input.GetDevice();
        auto unitDataWrapper = model.GetUnitDataWrapper(m_unitKey);

        Util::TensorData weight = unitDataWrapper.TensorDataMap["weight"];
        Util::TensorData transposedWeight =
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