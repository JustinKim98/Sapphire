// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UNIT_UTILS_HPP
#define SAPPHIRE_UNIT_UTILS_HPP
#include <type_traits>
#include <optional>
#include <Sapphire/tensor/TensorDescriptor.hpp>


namespace Sapphire::Util
{
template <typename T>
void ChangeTensorDataDimension(int dimension, T& tensorData)
{
    static_assert(std::is_same_v<TensorUtil::TensorData, T>);
    tensorData.TensorShape.Expand(dimension);
    tensorData.TensorShape.Shrink(dimension);
    tensorData.TensorShape[0] *= tensorData.BatchSize;
}

template <typename T, typename... Ts>
void ChangeTensorDataDimension(int dimension, T& tensorData, Ts&... params)
{
    static_assert(std::is_same_v<TensorUtil::TensorData, T>);
    tensorData.TensorShape.Expand(dimension);
    tensorData.TensorShape.Shrink(dimension);
    tensorData.TensorShape[0] *= tensorData.BatchSize;

    ChangeTensorDataDimension(dimension, params...);
}

template <std::size_t I = 0, typename... Tp>
inline void AddOutputHistory(SharedPtr<BackProp::BackPropWrapper> wrapper,
                             std::tuple<Tp...> t)
{
    if constexpr (I < sizeof...(Tp))
    {
        std::get<I>(t)->AppendOutputHistory(wrapper, I);
        AddOutputHistory<I + 1, Tp...>(wrapper, t);
    }
}

template <std::size_t I = 0, typename... Tp>
inline void AddOperandHistory(
    TensorUtil::TensorDescriptor* input, std::tuple<Tp...> t)
{
    if constexpr (I < sizeof...(Tp))
    {
        input->AppendOperandHistory(std::get<I>(t)->GetKey());
        AddOperandHistory<I + 1, Tp...>(input, t);
    }
}


//! Saves history for tensors used in the unit
//! Adds OperandHistory for inputs
//! Adds OutputHistory for outputs
//! \tparam inputIdx : current index of the input parameter
//! \tparam InputTs : packed parameter types for inputs
//! \tparam OutputTs : packed parameters types for outputs
//! \param wrapper : SharedPtr to the backPropWrapper for this unit
//! \param inputs : Tuple of pointers of TensorUtil::TensorDescriptor* of inputs
//! \param outputs : Tuple of pointers of TensorUtil::TensorDescriptor* of outputs
template <std::size_t inputIdx = 0, typename... InputTs, typename... OutputTs>
inline void SaveHistory(
    SharedPtr<BackProp::BackPropWrapper> wrapper, std::tuple<InputTs...> inputs,
    std::tuple<OutputTs...> outputs)
{
    if constexpr (inputIdx == sizeof...(InputTs))
    {
        AddOutputHistory(wrapper, outputs);
    }
    else
    {
        AddOperandHistory(std::get<inputIdx>(inputs), outputs);
        SaveHistory<inputIdx + 1>(wrapper, inputs, outputs);
    }
}

template <typename T>
inline bool CheckBatchSizeEquality(unsigned int batchSize, const T& param)
{
    return batchSize == param.GetBatchSize();
}

template <typename T, typename... Ts>
inline bool CheckBatchSizeEquality(unsigned int batchSize, const T& param,
                                   const Ts&... params)
{
    if (batchSize == param.GetBatchSize())
        return CheckBatchSizeEquality(batchSize, params...);
    return false;
}

template <typename T, typename ...Ts>
inline bool CheckBatchSizeEquality(const T& param, const Ts&... params)
{
    return CheckBatchSizeEquality(param.GetBatchSize(), params...);
}

template <typename T>
inline bool CheckDeviceEquality(const Device& device, const T& param)
{
    return device == param.GetDevice();
}

template <typename T, typename... Ts>
inline bool CheckDeviceEquality(const Device& device, const T& param,
                                const Ts&... params)
{
    if (device == param.GetDevice())
        return CheckDeviceEquality(device, params...);
    return false;
}

template <typename T, typename... Ts>
inline bool CheckDeviceEquality(const T& param, const Ts&... params)
{
    return CheckDeviceEquality(param.GetDevice(), params...);
}


inline std::optional<Shape> GetBroadcastedShape(const Shape& shapeA,
                                                const Shape& shapeB)
{
    int dimA = static_cast<int>(shapeA.Dim()) - 1;
    int dimB = static_cast<int>(shapeB.Dim()) - 1;
    std::vector<unsigned int> outputShapeVector(dimA > dimB ? dimA : dimB);
    int dimOut = static_cast<int>(outputShapeVector.size()) - 1;

    while (dimA >= 0 && dimB >= 0)
    {
        if (shapeA.At(dimA) == shapeB.At(dimB))
            outputShapeVector[dimOut] = shapeA.At(dimA);
        else if (shapeA.At(dimA) == 1)
            outputShapeVector[dimOut] = shapeB.At(dimB);
        else if (shapeB.At(dimB) == 1)
            outputShapeVector[dimOut] = shapeA.At(dimA);
        else
            return {};

        dimA -= 1;
        dimB -= 1;
        dimOut -= 1;
    }

    return Shape(outputShapeVector);
}
}

#endif
