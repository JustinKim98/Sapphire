// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/MSEBackward.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/util/SharedPtr.hpp>

namespace Sapphire::NN::Loss
{
Tensor MSE(const Tensor& input, const Tensor& label)
{
    Model& model = ModelManager::GetCurrentModel();

    auto& xDesc = model.GetDescriptor(input.TensorDescriptorKey());
    auto& labelDesc = model.GetDescriptor(label.TensorDescriptorKey());
    const auto yDescKey = model.RegisterTensorDescriptor(
        Shape({ 1 }), xDesc.GetType(),
        xDesc.GetDevice());
    auto& yDesc = model.GetDescriptor(yDescKey);

    TensorUtil::TensorData temp(
        input.GetForwardDataShape(), xDesc.GetType(),
        xDesc.GetDevice(), xDesc.GetBatchSize());

    auto x = xDesc.GetForwardData();
    auto lb = labelDesc.GetForwardData();
    auto y = yDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    auto wrapper =
        Util::SharedPtr<BackProp::MSEBackward>::Make(dx, x, lb);
    Util::SaveHistory(wrapper, std::make_tuple(&xDesc, &labelDesc),
                      std::make_tuple(&yDesc));

    Util::ChangeTensorDataDimension(1, x, lb, y, dx, temp);

    Compute::Sub(temp, lb, x);
    Compute::Pow(temp, temp, 2.0f);
    Compute::Mean(y, temp);
    return Tensor(yDescKey);
}
} // namespace Sapphire::NN::Loss
