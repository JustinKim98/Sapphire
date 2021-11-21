// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Loss/CrossEntropy.hpp>
#include <Sapphire/operations/Backward/CrossEntropyBackward.hpp>
#include <Sapphire/compute/LossOps.hpp>
#include <Sapphire/util/UnitUtils.hpp>


namespace Sapphire::NN::Loss
{
Tensor CrossEntropy(const Tensor& input, const Tensor& label)
{
    static int unitIdCount = 0;
    auto mode = input.Mode();
    if (!Util::CheckModeEquality(mode, label))
        throw std::invalid_argument(
            "NN::Loss::CrossEntropy - Device mode inequality");

    Model& model = ModelManager::CurModel();

    auto& xDesc = model.GetDescriptor(input.TensorDescriptorKey());
    auto& labelDesc = model.GetDescriptor(label.TensorDescriptorKey());

    const int batchSize = xDesc.GetShape().GetNumUnits(1);

    const auto yDescKey = model.RegisterTensorDescriptor(
        Shape({ batchSize, 1 }), xDesc.GetType(), xDesc.GetCudaDevice());
    auto& yDesc = model.GetDescriptor(yDescKey);
    yDesc.SetMode(mode);

    TensorUtil::TensorData diff(input.GetShape(), xDesc.GetType(),
                                xDesc.GetCudaDevice());

    auto xData = xDesc.GetForwardData();
    auto dxData = xDesc.GetBackwardData();
    auto labelData = labelDesc.GetForwardData();
    auto yData = yDesc.GetForwardData();

    auto* wrapper = new BackProp::CrossEntropyBackward(
        "CrossEntropy" + std::to_string(unitIdCount++), dxData, labelData);
    Util::SaveHistory(wrapper, std::make_tuple(&xDesc, &labelDesc),
                      std::make_tuple(&yDesc));

    Util::ChangeTensorDataDimension(2, xData, labelData, yData, dxData);
    Compute::CrossEntropy(yData, xData, labelData);

    return Tensor(yDescKey);
}
}
