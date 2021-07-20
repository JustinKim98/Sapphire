// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/LinearBackward.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::NN
{
Linear::Linear(unsigned int inputFeatureSize, unsigned int outputFeatureSize,
               const Device& device, bool bias, bool isSparse)
    : m_outputs(outputFeatureSize),
      m_bias(bias)
{
    auto& currentModel = ModelManager::GetCurrentModel();
    const Type type = isSparse ? Type::Sparse : Type::Dense;
    UnitDataWrapper wrapper;
    wrapper.TensorDataMap["weight"] = TensorUtil::TensorData(
        Shape({ inputFeatureSize, outputFeatureSize }), type, device, 1);

    wrapper.TensorDataMap["bias"] =
        TensorUtil::TensorData(Shape({ outputFeatureSize }), type, device, 1);

    //! Initialize bias and weight
    m_unitKey = currentModel.RegisterUnitDataWrapper(wrapper);
}

Tensor Linear::operator()(const Tensor& tensor) const
{
    auto& model = ModelManager::GetCurrentModel();
    auto unitDataWrapper = model.GetUnitDataWrapper(m_unitKey);

    TensorUtil::TensorDescriptor& xDesc =
        model.GetDescriptor(tensor.TensorDescriptorKey());

    Shape shapeInput = xDesc.ForwardData.TensorShape;
    const unsigned int batchSize = xDesc.ForwardData.BatchSize;
    const Type type = xDesc.ForwardData.GetType();
    const Device device = xDesc.ForwardData.GetDevice();
    const Shape outputShape({ m_outputs });

    const auto yKey = model.RegisterTensorDescriptor(outputShape, type, device,
        batchSize, true);
    auto& yDesc = model.GetDescriptor(yKey);

    Compute::Gemm(yDesc.ForwardData, xDesc.ForwardData,
                  unitDataWrapper.TensorDataMap["weight"],
                  unitDataWrapper.TensorDataMap["bias"]);

    auto backPropWrapper = std::make_unique<BackProp::LinearBackProp>(
        xDesc.ForwardData, xDesc.BackwardData, yDesc.BackwardData,
        m_unitKey);

    //! Append operand history to the inputDescriptor
    xDesc.AppendOperandHistory(yKey);
    //! Append output history to the output descriptor
    yDesc.AppendOutputHistory(std::move(backPropWrapper), true);

    return Tensor(yKey);
}
} // namespace Sapphire::NN
