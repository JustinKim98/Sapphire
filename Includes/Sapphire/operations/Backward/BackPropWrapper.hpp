// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_BACKPROP_WRAPPER_HPP
#define SAPPHIRE_BACKPROP_WRAPPER_HPP

#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/operations/optimizers/Optimizer.hpp>
#include <functional>
#include <memory>

namespace Sapphire::BackProp
{
//! todo : BackPropWrapper can be shared between objects and should backProp
//! when it is available only

//! This class is responsible for
//! 1. Storing the required data for back propagation
//! 2. Checking availability and invoking back propagation
//! This class is shared between tensorDescriptors that has been created from
//! the same operation
class BackPropWrapper
{
public:
    BackPropWrapper() = default;
    virtual ~BackPropWrapper() = default;

    explicit BackPropWrapper(
        std::vector<TensorUtil::TensorData> gradientOutputs,
        std::vector<TensorUtil::TensorData> gradientInputs,
        std::vector<TensorUtil::TensorData> trainableData,
        std::weak_ptr<Optimizer::Optimizer> optimizer, int unitKey)
        : m_unitKey(unitKey),
          m_gradientOutputs(std::move(gradientOutputs)),
          m_gradientInputs(std::move(gradientInputs)),
          m_trainableData(std::move(trainableData)),
          m_optimizer(std::move(optimizer))
    {
    }

    explicit BackPropWrapper(
        std::vector<TensorUtil::TensorData> gradientOutputs,
        std::vector<TensorUtil::TensorData> gradientInputs)
        : m_gradientOutputs(std::move(gradientOutputs)),
          m_gradientInputs(std::move(gradientInputs))
    {
    }

    BackPropWrapper(const BackPropWrapper& backPropWrapper) = default;
    BackPropWrapper(BackPropWrapper&& backPropWrapper) noexcept = default;
    BackPropWrapper& operator=(const BackPropWrapper& backPropWrapper) = default
    ;
    BackPropWrapper& operator=(BackPropWrapper&& backPropWrapper) noexcept
    = default;

    [[nodiscard]] const std::vector<TensorUtil::TensorData>&
    GetOutputTensorDataVector() const
    {
        return m_gradientOutputs;
    }

    //! InvokeBackProp should check if BackPropWrapper is ready before invoking back propagation
    virtual bool InvokeBackProp(const TensorUtil::TensorData& input) = 0;

protected:
    int m_unitKey = -1;
    //! Vector of tensorData that should give its output
    std::vector<TensorUtil::TensorData> m_gradientOutputs;
    std::vector<TensorUtil::TensorData> m_gradientInputs;
    std::vector<TensorUtil::TensorData> m_trainableData;
    std::weak_ptr<Optimizer::Optimizer> m_optimizer;
    std::unordered_map<std::string, TensorUtil::TensorData> m_savedTensorMap;
};
} // namespace Sapphire::BackProp

#endif
