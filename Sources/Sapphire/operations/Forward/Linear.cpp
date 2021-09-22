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
Linear::Linear(int inputFeatureSize, int outputFeatureSize,
               Util::SharedPtr<Optimizer::Optimizer> optimizer,
               std::unique_ptr<Initialize::Initializer> weightInitializer,
               std::unique_ptr<Initialize::Initializer> biasInitializer,
               CudaDevice device, bool isSparse)
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

    auto weight = TensorUtil::TensorData(
        Shape({ inputFeatureSize, outputFeatureSize }), type, m_device);
    auto bias =
        TensorUtil::TensorData(Shape({ outputFeatureSize }), type, m_device);
    auto transposedWeight = TensorUtil::TensorData(
        Shape({ outputFeatureSize, inputFeatureSize }), type, m_device);

    weightInitializer->operator()(weight);
    biasInitializer->operator()(bias);

    //! Synchronize data between host and device
    if (m_device.GetID() >= 0)
    {
        weight.ToCuda();
        bias.ToCuda();
        transposedWeight.ToCuda();
        weight.SetMode(DeviceType::Cuda);
        bias.SetMode(DeviceType::Cuda);
        transposedWeight.SetMode(DeviceType::Cuda);
    }

    m_trainableDataMap["weight"] = std::move(weight);
    m_trainableDataMap["bias"] = std::move(bias);
    m_mutableDataMap["transposedWeight"] = std::move(transposedWeight);
}

Tensor Linear::operator()(Tensor& xTensor)
{
    auto mode = xTensor.Mode();
    auto& model = ModelManager::GetCurrentModel();

    auto& xDesc =
        model.GetDescriptor(xTensor.TensorDescriptorKey());
    m_checkArguments({ &xDesc });
    const auto yKey = m_registerOutputTensor(xDesc);
    auto& yDesc = model.GetDescriptor(yKey);
    yDesc.SetMode(mode);

    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();

    auto weight = m_trainableDataMap.at("weight");
    auto bias = m_trainableDataMap.at("bias");
    auto transposedWeight =
        m_mutableDataMap["transposedWeight"];

    //! Change the dimension of the data to match the requirements
    Util::ChangeTensorDataDimension(2, x, dx, y, dy);

    Compute::Initialize::Zeros(y);
    Compute::Transpose(transposedWeight, weight);
    //! Bias is broadcasted internally
    Compute::Gemm(y, x, transposedWeight, bias);

    auto backPropWrapper =
        Util::SharedPtr<BackProp::LinearBackProp>::Make(
            dx, dy, weight, bias, x.CreateCopy(), m_optimizer,
            x.GetBatchSize(2));
    SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                std::make_tuple(&yDesc));
    return Tensor(yKey);
}

int Linear::m_registerOutputTensor(
    const TensorUtil::TensorDescriptor& xDesc) const
{
    auto& model = ModelManager::GetCurrentModel();
    const auto x = xDesc.GetForwardData();
    const Shape xShape = xDesc.GetShape();
    Shape yShape = xShape;
    yShape[yShape.Dim() - 1] = m_outputs;
    const auto yKey = model.RegisterTensorDescriptor(
        yShape, xDesc.GetType(), xDesc.GetDevice());
    return yKey;
}

void Linear::m_checkArguments(
    std::vector<TensorUtil::TensorDescriptor*> arguments) const
{
    const auto input = arguments.at(0);
    if (input->GetShape().Cols() != m_inputs)
        throw std::invalid_argument("NN::Linear - Shape mismatch");
    if (input->GetDevice() != m_device)
        throw std::invalid_argument("NN::Linear - Device mismatch");
}
} // namespace Sapphire::NN
