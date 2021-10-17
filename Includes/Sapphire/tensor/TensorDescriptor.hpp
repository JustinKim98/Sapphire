// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_UTIL_TENSORDESCRIPTOR_DECL_HPP
#define Sapphire_UTIL_TENSORDESCRIPTOR_DECL_HPP

#include <Sapphire/operations/Backward/BackPropWrapper.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <list>
#include <memory>
#include <mutex>

namespace Sapphire::TensorUtil
{
//! TensorDescriptor stores metaData of the tensor
//! There can be more than one tensor that references to one tensorData
//! All public functions in the TensorDescriptor is guaranteed to be thread safe
//! TensorDescriptor should not be accessible from the user interface directly
class TensorDescriptor
{
public:
    TensorDescriptor() = default;
    TensorDescriptor(const Shape& shape, Type type, const CudaDevice& device,
                     int key, bool preserve = false);

    ~TensorDescriptor() = default;

    TensorDescriptor(const TensorDescriptor& tensorData) = delete;
    TensorDescriptor(TensorDescriptor&& tensorData) noexcept;
    TensorDescriptor& operator=(const TensorDescriptor& tensorData) = delete;
    TensorDescriptor& operator=(TensorDescriptor&& tensorDesc) noexcept;

    //! Gets shallow copy of the forward TensorData
    [[nodiscard]] TensorData GetForwardData() const;
    //! Gets shallow copy of the backward TensorData
    [[nodiscard]] TensorData GetBackwardData() const;
    //! Gets batch size of internal TensorData
    [[nodiscard]] unsigned int GetBatchSize() const;

    [[nodiscard]] Shape GetShape() const;
    [[nodiscard]] CudaDevice GetDevice() const;
    [[nodiscard]] CudaDevice GetCudaDevice() const;
    [[nodiscard]] Type GetType() const;

    void Reshape(Shape shape);

    //! Moves internal TensorData to cuda
    void ToCuda();

    //! Moves internal TensorData to host
    void ToHost();

    //! Gets current mode of the descriptor
    [[nodiscard]] DeviceType Mode() const;

    //! Sets the mode of the descriptor
    void SetMode(DeviceType deviceType);

    //! Initializes backward data to zero
    void InitGradient();

    //! Add unit m_key if unit was used as output or flow-through type
    //! \param backPropWrapperKey : backPropWrapper for starting back propagation on this tensor
    //! \param location : Forward output of this tensorDescriptor is preserved
    //! if true
    void AppendOutputHistory(
        int backPropWrapperKey, int location);

    //! Add unit key if unit was used as operand only
    //! \param tensorDescKey : m_key of the tensor that this tensor should receive
    //! gradient from
    void AppendOperandHistory(int tensorDescKey);

    //! Removes the gradient input key
    //! The last history must not be output
    //! \param tensorDescKey : key of the operand target tensor to remove
    void RemoveOperand(int tensorDescKey);

    //! Removes last history from the history list if it is operand history and history list is not empty
    void PopIfOperandHistory();

    //! Removes the last history if not empty
    void PopOutputHistory();

    //! Returns whether this tensorDescriptor is trainable
    //! \return : True if gradient is required false otherwise
    [[nodiscard]] bool IsTrainable() const
    {
        return m_trainable;
    }

    //! Checks if next operation is output unit in back propagation
    //! \return : true if ready false otherwise
    [[nodiscard]] bool IsBackPropReady() const;

    std::pair<int, int>
    GetBackPropWrapperKeyFromLastHistory()
    {
        const auto& history = m_history.back();
        return std::make_pair(history.BackPropWrapperKey, history.Location);
    }

    bool HasHistory()
    {
        return !m_history.empty();
    }

    [[nodiscard]] int GetKey() const
    {
        return m_key;
    }

private:
    TensorData m_forwardData;
    TensorData m_backwardData;

    //! This describes history of the tensorData
    //! As tensorData is used in unit function as an operand or input/output.
    //! It is stored using this struct
    struct History
    {
        //! This constructor creates output history, where tensor was newly created
        //! This kind of history will invoke backPropWrapper
        explicit History(int backPropWrapperKey,
                         int location)
            : IsOutput(true),
              Location(location),
              BackPropWrapperKey(backPropWrapperKey)
        {
        }

        //! This constructor creates operand history, where tensor gave its values to other units
        //! to be used in other operations in forward pass
        //! Tensor will have to receive gradients from all operations it gave out values in the backward pass
        History()
            : IsOutput(false)
        {
        }

        ~History() = default;

        History(History&& history) noexcept
            : IsOutput(history.IsOutput),
              Location(history.Location),
              GradientInputTensorKeyList(
                  std::move(history.GradientInputTensorKeyList))
        {
            if (IsOutput)
                BackPropWrapperKey = std::move(history.BackPropWrapperKey);
        }

        History(const History& history) = delete;

        History& operator=(History&& history) noexcept
        {
            IsOutput = history.IsOutput;
            Location = history.Location;
            if (IsOutput)
                BackPropWrapperKey = history.BackPropWrapperKey;
            GradientInputTensorKeyList =
                std::move(history.GradientInputTensorKeyList);
            return *this;
        }

        History& operator=(const History& history) = delete;

        //! Add tensor descriptor key to receive the gradient input
        void AddOperand(int tensorDescKey)
        {
            GradientInputTensorKeyList.emplace_back(tensorDescKey);
        }

        void RemoveOperand(int tensorDescKey)
        {
            const auto it = std::find(
                GradientInputTensorKeyList.begin(),
                GradientInputTensorKeyList.end(), tensorDescKey);

            if (it == GradientInputTensorKeyList.end())
                throw std::runtime_error(
                    "History::RemoveOperand - given "
                    "tensorDescKey was not found in the operand history");

            GradientInputTensorKeyList.erase(it);
        }

        bool IsOutput;
        //! Location specifies which index that tensor was created
        int Location = 0;

        int BackPropWrapperKey;
        //! List of the units that was as operand
        std::list<int> GradientInputTensorKeyList;
    };

    //! m_key to identify tensor data
    int m_key = -1;
    unsigned int m_batchSize = 0;
    bool m_trainable = true;

    std::list<History> m_history;
};
} // namespace Sapphire::TensorUtil

#endif
