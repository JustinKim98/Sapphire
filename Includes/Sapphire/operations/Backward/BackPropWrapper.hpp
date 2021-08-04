// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_BACKPROP_WRAPPER_HPP
#define SAPPHIRE_BACKPROP_WRAPPER_HPP

#include <algorithm>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/operations/optimizers/Optimizer.hpp>
#include <functional>
#include <memory>

namespace Sapphire::BackProp
{
//! BackPropWrapper can be shared between objects

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
        std::vector<TensorUtil::TensorData> dxVector,
        std::vector<TensorUtil::TensorData> dyVector,
        std::vector<TensorUtil::TensorData> trainableData,
        std::vector<TensorUtil::TensorData> constants,
        std::vector<TensorUtil::TensorData> mutables,
        Util::SharedPtr<Optimizer::Optimizer> optimizer)
        : m_dxVector(std::move(dxVector)),
          m_dyVector(std::move(dyVector)),
          m_trainableData(std::move(trainableData)),
          m_constants(std::move(constants)),
          m_mutables(std::move(mutables)),
          m_optimizer(std::move(optimizer)),
          m_receivedGradients(dyVector.size(), false)
    {
    }

    explicit BackPropWrapper(
        std::vector<TensorUtil::TensorData> dxVector,
        std::vector<TensorUtil::TensorData> dyVector,
        std::vector<TensorUtil::TensorData> constants,
        std::vector<TensorUtil::TensorData> mutables)
        : m_dxVector(std::move(dxVector)),
          m_dyVector(std::move(dyVector)),
          m_constants(std::move(constants)),
          m_mutables(std::move(mutables)),
          m_receivedGradients(dyVector.size(), false)

    {
    }

    explicit BackPropWrapper(
        std::vector<TensorUtil::TensorData> dxVector,
        std::vector<TensorUtil::TensorData> dyVector)
        : m_dxVector(std::move(dxVector)),
          m_dyVector(std::move(dyVector)),
          m_receivedGradients(dyVector.size(), false)
    {
    }


    BackPropWrapper(const BackPropWrapper& backPropWrapper) = default;
    BackPropWrapper(BackPropWrapper&& backPropWrapper) noexcept = default;
    BackPropWrapper& operator=(const BackPropWrapper& backPropWrapper) = delete;
    BackPropWrapper& operator=(BackPropWrapper&& backPropWrapper) noexcept
    = delete;

    [[nodiscard]] std::vector<int>
    GetGradientOutputDescriptorKeys() const
    {
        std::vector<int> tensorKeys(m_dxVector.size());

        for (std::size_t i = 0; i < m_dyVector.size(); ++i)
            tensorKeys[i] = m_dxVector[i].GetDescriptorKey();

        return tensorKeys;
    }

    //! InvokeBackPropIfReady checks if BackPropWrapper is ready before invoking back propagation
    //! \param location : The id of the parameter. Id always starts from 0 with the first parameter (from the left)
    bool InvokeBackPropIfReady(int location)
    {
        if (m_isReady(location))
        {
            m_runBackProp();
            return true;
        }
        return false;
    }

protected:
    bool m_isReady(int location)
    {
        if (m_receivedGradients.at(location))
            throw std::runtime_error(
                "BackProp::BackPropWrapper::m_isReady - Received gradient two "
                "times from same location");

        m_receivedGradients.at(location) = true;
        return std::all_of(m_receivedGradients.begin(),
                           m_receivedGradients.end(),
                           [](auto x) { return x; });
    }

    virtual void m_runBackProp() = 0;

    //! Vector of tensorData that should give its output
    std::vector<TensorUtil::TensorData> m_dxVector;
    const std::vector<TensorUtil::TensorData> m_dyVector;
    std::vector<TensorUtil::TensorData> m_trainableData;
    const std::vector<TensorUtil::TensorData> m_constants;
    std::vector<TensorUtil::TensorData> m_mutables;
    Util::SharedPtr<Optimizer::Optimizer> m_optimizer;
    std::vector<bool> m_receivedGradients;
    //! Data saved in m_constants should not be modified
};
} // namespace Sapphire::BackProp

#endif
