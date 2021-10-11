// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/MSEBackward.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <Sapphire/util/UnitUtils.hpp>

namespace Sapphire::NN::Loss
{
Tensor MSE(const Tensor& input, const Tensor& label)
{
    auto mode = input.Mode();
    if (!Util::CheckModeEquality(mode, label))
        throw std::invalid_argument("NN::Loss::MSE - Device mode inequality");
    Model& model = ModelManager::CurModel();

    auto& xDesc = model.GetDescriptor(input.TensorDescriptorKey());
    auto& labelDesc = model.GetDescriptor(label.TensorDescriptorKey());

    const auto yDescKey = model.RegisterTensorDescriptor(
        Shape({ 1 }), xDesc.GetType(),
        xDesc.GetCudaDevice());
    auto& yDesc = model.GetDescriptor(yDescKey);
    yDesc.SetMode(mode);

    TensorUtil::TensorData temp(
        input.GetShape(), xDesc.GetType(),
        xDesc.GetCudaDevice());
    temp.SetMode(mode);

    auto xData = xDesc.GetForwardData();
    auto labelData = labelDesc.GetForwardData();
    auto yData = yDesc.GetForwardData();
    auto dxData = xDesc.GetBackwardData();
    auto* wrapper = new BackProp::MSEBackward(dxData, xData, labelData);
    Util::SaveHistory(wrapper, std::make_tuple(&xDesc, &labelDesc),
                      std::make_tuple(&yDesc));

    Util::ChangeTensorDataDimension(1, xData, labelData, yData, dxData, temp);

    Compute::Sub(temp, labelData, xData);
    Compute::Pow(temp, temp, 2.0f);
    Compute::Mean(yData, temp, 0);
    return Tensor(yDescKey);
}
} // namespace Sapphire::NN::Loss
