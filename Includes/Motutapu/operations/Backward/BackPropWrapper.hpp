// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_BACKPROPWRAPPER_HPP
#define MOTUTAPU_BACKPROPWRAPPER_HPP

#include <Motutapu/compute/dense/cuda/Basic.cuh>
#include <Motutapu/compute/dense/naive/NaiveBasic.hpp>
#include <Motutapu/tensor/TensorData.hpp>
#include <functional>
#include <list>

namespace Motutapu::BackProp
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
        std::vector<TensorUtil::TensorData> gradientInputs)
        : m_gradientOutputs(std::move(gradientOutputs)),
          m_gradientInputs(std::move(gradientInputs))
    {
    }

    explicit BackPropWrapper(
        std::vector<TensorUtil::TensorData> gradientOutputs,
        std::vector<TensorUtil::TensorData> gradientInputs, int unitKey)
        : m_unitKey(unitKey),
          m_gradientOutputs(std::move(gradientOutputs)),
          m_gradientInputs(std::move(gradientInputs))
    {
    }

    [[nodiscard]] const std::vector<TensorUtil::TensorData>&
    GetOutputTensorKeys() const
    {
        return m_gradientOutputs;
    }

    //! todo : Copy required save data inside BackPropWrapper
    //! todo : Backward will only do its job when all inputs are provided
    //! Invokes back propagation if ready
    virtual bool InvokeBackProp(const TensorUtil::TensorData& input) = 0;

 protected:
    int m_unitKey = -1;
    //! Vector of tensorData that should give its output
    std::vector<TensorUtil::TensorData> m_gradientOutputs;
    std::vector<TensorUtil::TensorData> m_gradientInputs;
    std::unordered_map<std::string, TensorUtil::TensorData> m_savedTensorMap;
};
}  // namespace Motutapu::BackProp

#endif
