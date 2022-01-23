// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UNIT_UTILS_HPP
#define SAPPHIRE_UNIT_UTILS_HPP

#include <Sapphire/tensor/TensorDescriptor.hpp>
#include <Sapphire/Model.hpp>
#include <type_traits>
#include <optional>

namespace Sapphire::Util
{
template <typename T>
void ChangeTensorDataDimension(int dimension, T& tensorData)
{
    static_assert(std::is_same_v<TensorUtil::TensorData, T>);
    auto newShape = tensorData.GetShape();
    newShape.Expand(dimension);
    newShape.Shrink(dimension);
    tensorData.Reshape(newShape);
}

template <typename T, typename... Ts>
void ChangeTensorDataDimension(int dimension, T& tensorData, Ts&... params)
{
    static_assert(std::is_same_v<TensorUtil::TensorData, T>);
    auto newShape = tensorData.GetShape();
    newShape.Expand(dimension);
    newShape.Shrink(dimension);
    tensorData.Reshape(newShape);
    ChangeTensorDataDimension(dimension, params...);
}

inline int GetMatchingDim(std::vector<Shape> shapes)
{
    if (shapes.empty())
        throw std::runtime_error("Util::GetMatchingDim - shapes is empty");
    int curDimFromLast = -1;
    bool match = true;

    while (match)
    {
        if (shapes.empty())
            break;

        if (shapes.at(0).Dim() + curDimFromLast < 0)
        {
            break;
        }
        const int shapeDim = shapes.at(0).At(curDimFromLast);
        for (const auto& shape : shapes)
        {
            if (shape.Dim() + curDimFromLast < 0)
            {
                match = false;
                break;
            }
            if (shapeDim != shape.At(curDimFromLast))
            {
                match = false;
                break;
            }
        }

        if (match)
            curDimFromLast -= 1;
    }

    return -(curDimFromLast + 1);
}

template <std::size_t I = 0, typename... Tp>
void AddOutputHistory(int backPropWrapperKey,
                      std::tuple<Tp...> t)
{
    if constexpr (I < sizeof...(Tp))
    {
        std::get<I>(t)->AppendOutputHistory(backPropWrapperKey, I);
        AddOutputHistory<I + 1, Tp...>(backPropWrapperKey, t);
    }
}

template <std::size_t I = 0, typename... Tp>
void AddOperandHistory(
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
//! After adding all operand history, Adds all OutputHistory for outputs
//! \tparam inputIdx : current index of the input parameter
//! \tparam InputTs : packed parameter types for inputs
//! \tparam OutputTs : packed parameters types for outputs
//! \param wrapper : SharedPtr to the backPropWrapper for this unit
//! \param inputs : Tuple of pointers of TensorUtil::TensorDescriptor* of inputs
//! \param outputs : Tuple of pointers of TensorUtil::TensorDescriptor* of outputs
template <std::size_t inputIdx = 0, typename... InputTs, typename... OutputTs>
void SaveHistory(BackProp::BackPropWrapper* wrapper,
                 std::tuple<InputTs...> inputs,
                 std::tuple<OutputTs...> outputs)
{
    if constexpr (inputIdx == sizeof...(InputTs))
    {
        const auto backPropWrapperKey = ModelManager::CurModel().
            RegisterBackPropWrapper(wrapper);
        AddOutputHistory(backPropWrapperKey, outputs);
    }
    else
    {
        AddOperandHistory(std::get<inputIdx>(inputs), outputs);
        SaveHistory<inputIdx + 1>(wrapper, inputs, outputs);
    }
}

template <typename T, typename... Ts>
bool CheckDeviceEquality(const T& paramA, const T& paramB)
{
    if (const bool typeMatch = paramA.Mode() == paramB.Mode(); !typeMatch)
        return false;

    if (paramA.Mode() == ComputeMode::Cuda &&
        paramA.GetDeviceInfo() != paramB.GetDeviceInfo())
        return false;

    return true;
}

template <typename T, typename... Ts>
bool CheckDeviceEquality(const T& paramA, const T& paramB,
                         const Ts&... params)
{
    if (const bool typeMatch = paramA.Mode() == paramB.Mode(); !typeMatch)
        return false;

    if (paramA.Mode() == ComputeMode::Cuda &&
        paramA.GetDeviceInfo() != paramB.GetDeviceInfo())
        return false;

    return CheckDeviceEquality(paramB, params...);
}


inline std::optional<Shape> GetBroadcastedShape(const Shape& shapeA,
                                                const Shape& shapeB,
                                                int requiredDim)
{
    int dimA = shapeA.Dim() - 1 - requiredDim;
    int dimB = shapeB.Dim() - 1 - requiredDim;
    std::vector<int> outputShapeVector(
        shapeA.Dim() > shapeB.Dim()
            ? shapeA.Dim()
            : shapeB.Dim());

    int dimOut = static_cast<int>(outputShapeVector.size()) - 1 - requiredDim;

    while (dimA >= 0 && dimB >= 0)
    {
        if (shapeA.At(dimA) == 1)
            outputShapeVector[dimOut] = shapeB.At(dimB);
        else if (shapeB.At(dimB) == 1 || shapeB.At(dimB) == shapeA.At(dimA))
            outputShapeVector[dimOut] = shapeA.At(dimA);
        else
            return {};

        dimA -= 1;
        dimB -= 1;
        dimOut -= 1;
    }

    while (dimOut >= 0)
    {
        if (dimA >= 0)
        {
            outputShapeVector[dimOut] = shapeA.At(dimA);
            dimA -= 1;
        }
        if (dimB >= 0)
        {
            outputShapeVector[dimOut] = shapeB.At(dimB);
            dimB -= 1;
        }
        dimOut -= 1;
    }

    return Shape(outputShapeVector);
}

template <typename TensorT>
bool CheckModeEquality(ComputeMode mode, TensorT tensor)
{
    return mode == tensor.Mode();
}

template <typename TensorT, typename ...TensorTs>
bool CheckModeEquality(ComputeMode mode, TensorT tensor,
                       TensorTs ... tensors)
{
    if (mode == tensor.Mode())
        return CheckModeEquality(mode, tensors...);
    else
        return false;
}
}

#endif
