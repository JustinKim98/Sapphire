// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_TENSORDESCRIPTOR_DECL_HPP
#define MOTUTAPU_UTIL_TENSORDESCRIPTOR_DECL_HPP

#include <Motutapu/tensor/TensorData.hpp>
#include <Motutapu/operations/Backward/BackPropWrapper.hpp>
#include <list>
#include <mutex>

namespace Motutapu::Util
{
//! TensorDescriptor stores real tensor data of the tensor
//! There can be more than one tensor that references to one tensorData
//! All public functions in the TensorDescriptor is guaranteed to be thread safe
//! TensorDescriptor should not be accessible from the user interface directly
template <typename T>
class TensorDescriptor
{
public:

    TensorDescriptor() = default;
    ~TensorDescriptor() = default;

    //! Create and allocate the tensor descriptor
    TensorDescriptor(Shape shape, Type type, Device device,
                     unsigned int batchSize);

    TensorDescriptor(const TensorDescriptor& tensorData) = delete;
    TensorDescriptor(TensorDescriptor&& tensorData) noexcept = default;
    TensorDescriptor& operator=(const TensorDescriptor& tensorData) = delete;
    TensorDescriptor& operator=(TensorDescriptor&& tensorData) noexcept
    = default;

    TensorData<T> ForwardData;
    TensorData<T> BackwardData;

    //! Key to identify tensor data
    int Key = -1;

    //! Add unit Key if unit was used as output or flow-through type
    //! \param wrapper : Wrapper for starting back propagation on this tensor
    //! \param saveOutput : Forward output of this tensorDescriptor is preserved if true
    void AppendOutputHistory(
        std::unique_ptr<BackProp::BackPropWrapper<T>> wrapper,
        bool saveOutput);

    //! Add unit key if unit was used as operand only
    //! \param tensorKey : Key of the tensor that this tensor should receive gradient from
    void AppendOperandHistory(int tensorKey);

    void RemoveGradientInputKey(int tensorKey);
    //! Removes last history from the history list
    void PopHistory();

    //! Create new tensor if last tensor required output saving
    [[nodiscard]] bool RequireOutputSaving() const
    {
        return m_requireOutputSaving;
    }

    //! Checks if next operation is output unit in back propagation
    //! Removes operand history if it is full
    //! \return : true if ready false otherwise
    bool IsBackPropReady()
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);

        if (m_history.empty())
            return false;

        if (m_history.back().IsOutput())
            return true;

        return false;
    }

    const std::unique_ptr<BackProp::BackPropWrapper<T>>& GetBackPropWrapper()
    {
        return m_history.back().Wrapper;
    }

private:

    //! This describes history of the tensorData
    //! As tensorData is used in unit function as an operand or input/output.
    //! It is stored using this struct
    struct History
    {
        History(std::unique_ptr<BackProp::BackPropWrapper<T>> wrapper)
            : IsOutput(true),
              Wrapper(std::move(wrapper))
        {
        }

        History()
            : IsOutput(false)
        {
        }

        void AddGradientInputTensorKey(int key)
        {
            GradientInputTensorKeys.emplace_back(key);
        }

        bool IsOutput;

        std::unique_ptr<BackProp::BackPropWrapper<T>> Wrapper;
        //! List of the units that was as operand
        std::list<int> GradientInputTensorKeys;
    };

    bool m_requireOutputSaving = false;

    std::list<History> m_history;
    //! mutex to make sure operations on the resources is synchronized
    std::recursive_mutex m_mtx;
};
} // namespace Motutapu::Util

#endif
