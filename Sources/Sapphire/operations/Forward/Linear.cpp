// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/LinearBackward.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::NN
{
Linear::Linear(unsigned int inputFeatureSize, unsigned int outputFeatureSize,
               std::shared_ptr<Optimizer::Optimizer> optimizer,
               Device device, bool isSparse)
    : m_outputs(outputFeatureSize),
      m_optimizer(std::move(optimizer)),
      m_device(std::move(device)),
      m_isSparse(isSparse)
{
    auto& currentModel = ModelManager::GetCurrentModel();
    const Type type = m_isSparse ? Type::Sparse : Type::Dense;
    UnitDataWrapper wrapper;
    wrapper.TensorDataMap["weight"] = TensorUtil::TensorData(
        Shape({ inputFeatureSize, outputFeatureSize }), type, m_device, 1);
    wrapper.TensorDataMap["bias"] =
        TensorUtil::TensorData(Shape({ outputFeatureSize }), type, m_device, 1);

    //! Initialize bias and weight
    m_unitWrapperKey = currentModel.AddUnitDataWrapper(wrapper);
}

Linear::~Linear()
{
    auto& currentModel = ModelManager::GetCurrentModel();
    currentModel.RemoveUnitDataWrapper(m_unitWrapperKey);
}

//! TODO : free the UnitDataWrapper in the destructor

Tensor Linear::operator()(const Tensor& input) const
{
    auto& model = ModelManager::GetCurrentModel();
    auto unitDataWrapper = model.GetUnitDataWrapper(m_unitWrapperKey);

    TensorUtil::TensorDescriptor& xDesc =
        model.GetDescriptor(input.TensorDescriptorKey());
    const auto yKey = m_registerOutputTensor(xDesc);
    auto& yDesc = model.GetDescriptor(yKey);

    auto x = xDesc.ForwardData;
    auto dx = xDesc.BackwardData;
    auto y = yDesc.ForwardData;
    auto dy = yDesc.BackwardData;

    //! Reshape the data to match the requirements
    x.TensorShape.Expand(2);
    dx.TensorShape.Expand(2);
    y.TensorShape.Expand(2);
    dy.TensorShape.Expand(2);
    x.TensorShape.Shrink(2);
    dx.TensorShape.Shrink(2);
    y.TensorShape.Shrink(2);
    dy.TensorShape.Shrink(2);

    auto& weight = unitDataWrapper.TensorDataMap["weight"];
    auto& bias = unitDataWrapper.TensorDataMap["bias"];

    TensorUtil::TensorData transposedWeight(weight.GetShape().GetTranspose(),
                                            weight.GetType(),
                                            weight.GetDevice(), 1);

    Compute::Initialize::Zeros(y);
    Compute::Transpose(transposedWeight, weight);
    Compute::Gemm(y, x, transposedWeight, bias);

    auto backPropWrapper = std::make_unique<BackProp::LinearBackProp>(
        dx, dy, weight, bias, x, m_optimizer, x.TensorShape.At(0),
        m_unitWrapperKey);

    xDesc.AppendOperandHistory(yKey);
    yDesc.AppendOutputHistory(std::move(backPropWrapper), 0);

    return Tensor(yKey);
}

int Linear::m_registerOutputTensor(
    const TensorUtil::TensorDescriptor& xDesc) const
{
    auto& model = ModelManager::GetCurrentModel();
    const auto x = xDesc.ForwardData;
    const Shape shapeInput = x.TensorShape;
    Shape outputShape = shapeInput;
    outputShape[outputShape.Dim() - 1] = m_outputs;
    const auto yKey = model.RegisterTensorDescriptor(
        outputShape, x.GetType(), x.GetDevice(), x.BatchSize, true);
    return yKey;
}
} // namespace Sapphire::NN
