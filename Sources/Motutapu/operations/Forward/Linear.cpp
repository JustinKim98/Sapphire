// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Model.hpp>
#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/operations/Backward/LinearBackward.hpp>
#include <Motutapu/operations/Forward/Linear.hpp>
#include <Motutapu/operations/Unit.hpp>
#include <Motutapu/tensor/TensorData.hpp>

namespace Motutapu::NN
{
//Linear::Linear(unsigned int inputFeatureSize, unsigned int outputFeatureSize,
//               const Device& device, bool bias, bool isSparse)
//    : m_outputs(outputFeatureSize), m_bias(bias)
//{
//    auto& currentModel = ModelManager::GetCurrentModel();
//    Type type = isSparse ? Type::Sparse : Type::Dense;
//    UnitDataWrapper wrapper;
//    wrapper.TensorDataMap["weight"] = TensorUtil::TensorData(
//        Shape({ inputFeatureSize, outputFeatureSize }), type, device, 1);
//
//    wrapper.TensorDataMap["TransposedWeight"] = TensorUtil::TensorData(
//        Shape({ outputFeatureSize, inputFeatureSize }), type, device, 1);
//
//    wrapper.TensorDataMap["bias"] =
//        TensorUtil::TensorData(Shape({ outputFeatureSize }), type, device, 1);
//
//    //! Initialize bias and weight
//    m_unitKey = currentModel.RegisterUnitDataWrapper(wrapper);
//}
//
//Tensor Linear::operator()(const Tensor& tensor) const
//{
//    auto& currentModel = ModelManager::GetCurrentModel();
//    auto unitDataWrapper = currentModel.GetUnitDataWrapper(m_unitKey);
//
//    TensorUtil::TensorDescriptor& descInput =
//        currentModel.GetDescriptor(tensor.TensorDescriptorKey());
//
//    auto shapeInput = descInput.ForwardData.TensorShape;
//    const auto batchSize = descInput.ForwardData.BatchSize;
//    const auto device = descInput.ForwardData.GetDevice();
//    const auto outputShape = Shape({ m_outputs });
//
//    TensorUtil::TensorDescriptor descOut(outputShape, m_type, device, batchSize,
//                                         true);
//    const auto outputKey = currentModel.RegisterTensorDescriptor(descOut);
//
//    Compute::Gemm(descOut.ForwardData, descInput.ForwardData,
//                  unitDataWrapper.TensorDataMap["weight"],
//                  unitDataWrapper.TensorDataMap["bias"]);
//
//    auto backPropWrapper =
//        std::make_unique<BackProp::LinearBackProp>(descInput.m_key, m_unitKey);
//
//    descInput.AppendOperandHistory(outputKey);
//    descOut.AppendOutputHistory(std::move(backPropWrapper), true);
//
//    return Tensor(outputShape, outputKey);
//}

}  // namespace Motutapu::NN