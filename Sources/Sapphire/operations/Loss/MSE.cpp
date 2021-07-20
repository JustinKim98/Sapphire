// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/MSEBackward.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <memory>

namespace Sapphire::NN::Loss
{
static Tensor MSE(const Tensor& x, const Tensor& label)
{
    Model& model = ModelManager::GetCurrentModel();

    auto& xDesc = model.GetDescriptor(x.TensorDescriptorKey());
    auto& labelDesc = model.GetDescriptor(label.TensorDescriptorKey());
    const auto yDescKey = model.RegisterTensorDescriptor(
        Shape({ 1 }), xDesc.ForwardData.GetType(),
        xDesc.ForwardData.GetDevice(), xDesc.GetBatchSize(), true);

    auto& yDesc = model.GetDescriptor(yDescKey);

    TensorUtil::TensorData temp(x.GetShape(), xDesc.ForwardData.GetType(),
                                xDesc.ForwardData.GetDevice(),
                                xDesc.GetBatchSize());

    Compute::Sub(temp, labelDesc.ForwardData, xDesc.ForwardData);
    Compute::Pow(temp, temp, 2.0f);

    Compute::Mean(yDesc.ForwardData, temp);

    auto backPropWrapper = std::make_unique<BackProp::MSEBackward>(
        xDesc.ForwardData, xDesc.BackwardData, labelDesc.ForwardData,
        yDesc.BackwardData);

    xDesc.AppendOperandHistory(yDesc.GetKey());
    labelDesc.AppendOperandHistory(yDesc.GetKey());
    yDesc.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(yDescKey);
}
} // namespace Sapphire::NN::Loss
