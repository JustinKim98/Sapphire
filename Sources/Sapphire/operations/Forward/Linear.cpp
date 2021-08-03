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
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/util/SharedPtr.hpp>

namespace Sapphire::NN
{
Linear::Linear(unsigned int inputFeatureSize, unsigned int outputFeatureSize,
               std::shared_ptr<Optimizer::Optimizer> optimizer,
               Device device, bool isSparse)
    : m_inputs(inputFeatureSize),
      m_outputs(outputFeatureSize),
      m_optimizer(std::move(optimizer)),
      m_device(std::move(device)),
      m_isSparse(isSparse)
{
    const Type type = m_isSparse ? Type::Sparse : Type::Dense;

    if (m_isSparse)
        throw std::invalid_argument(
            "NN::Linear - Sparse version not implemented");

    m_trainableDataMap["weight"] = TensorUtil::TensorData(
        Shape({ inputFeatureSize, outputFeatureSize }), type, m_device, 1);
    m_trainableDataMap["bias"] =
        TensorUtil::TensorData(Shape({ outputFeatureSize }), type, m_device, 1);
    m_mutableDataMap["transposedWeight"] = TensorUtil::TensorData(
        Shape({ outputFeatureSize, inputFeatureSize }), type, m_device, 1);
}


Tensor Linear::operator()(const Tensor& input)
{
    auto& model = ModelManager::GetCurrentModel();

    auto& xDesc =
        model.GetDescriptor(input.TensorDescriptorKey());
    const auto yKey = m_registerOutputTensor(xDesc);
    auto& yDesc = model.GetDescriptor(yKey);

    auto x = xDesc.GetForwardData().CreateCopy();
    auto dx = xDesc.GetBackwardData();
    auto y = yDesc.GetForwardData().CreateCopy();
    auto dy = yDesc.GetBackwardData();

    const auto& weight = m_trainableDataMap.at("weight");
    const auto& bias = m_trainableDataMap.at("bias");
    auto& transposedWeight =
        m_mutableDataMap["transposedWeight"];

    //! Reshape the data to match the requirements
    Util::ChangeTensorDataDimension(2, x, dx, y, dy);
    auto backPropWrapper = Util::SharedPtr<BackProp::LinearBackProp>::Make(
        dx, dy, weight, bias, x.CreateCopy(), m_optimizer,
        xDesc.GetShape().At(0));
    Util::SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&yDesc));

    Compute::Initialize::Zeros(y);
    Compute::Transpose(transposedWeight, weight);
    Compute::Gemm(y, x, transposedWeight, bias);
    return Tensor(yKey);
}

int Linear::m_registerOutputTensor(
    const TensorUtil::TensorDescriptor& xDesc) const
{
    auto& model = ModelManager::GetCurrentModel();
    const auto x = xDesc.GetForwardData();
    const Shape shapeInput = xDesc.GetShape();
    Shape outputShape = shapeInput;
    outputShape[outputShape.Dim() - 1] = m_outputs;
    const auto yKey = model.RegisterTensorDescriptor(
        outputShape, x.GetType(), x.GetDevice());
    return yKey;
}

bool Linear::m_checkArguments(
    std::vector<TensorUtil::TensorDescriptor> arguments)
{
    const auto& input = arguments.at(0);
    if (input.GetForwardData().GetShape().Cols() != m_inputs)
        throw std::invalid_argument("NN::Linear - Shape mismatch");
    if (input.GetForwardData().GetDevice() != m_device)
        throw std::invalid_argument("NN::Linear - Device mismatch");

    return true;
}
} // namespace Sapphire::NN
