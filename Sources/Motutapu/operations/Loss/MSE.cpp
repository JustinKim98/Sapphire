// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Model.hpp>
#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/operations/Loss/MSE.hpp>
#include <Motutapu/operations/Loss/MSEBackProp.hpp>
#include <memory>
#include <vector>

namespace Motutapu::NN::Loss
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

    auto backPropWrapper = std::make_unique<BackProp::MSEBackProp>(
        xDesc.ForwardData, xDesc.BackwardData, labelDesc.ForwardData,
        yDesc.BackwardData);

    xDesc.AppendOperandHistory(yDesc.GetKey());
    labelDesc.AppendOperandHistory(yDesc.GetKey());
    yDesc.AppendOutputHistory(std::move(backPropWrapper), false);
    // todo : Create mean compute function

    return Tensor(Shape({ 1 }), yDescKey);
}
}  // namespace Motutapu::NN::Loss