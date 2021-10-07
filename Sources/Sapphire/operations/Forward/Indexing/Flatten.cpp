// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/operations/Backward/Indexing/FlattenBackward.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/compute/ActivationOps.hpp>

namespace Sapphire::NN
{
void Flatten(const Tensor& xTensor)
{
    Model& model = ModelManager::GetCurrentModel();
    auto& xDesc = model.GetDescriptor(xTensor.TensorDescriptorKey());
    auto& yDesc = model.GetDescriptor(xTensor.TensorDescriptorKey());

    auto shape = xDesc.GetShape();

    auto xData = xDesc.GetForwardData();
    auto dxData = xDesc.GetBackwardData();
    auto yData = yDesc.GetForwardData();
    auto dyData = yDesc.GetBackwardData();
    auto* wrapper =
        new BackProp::FlattenBackward(dxData, dxData, shape);
    Util::SaveHistory(wrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&xDesc));
    xDesc.SetShape(shape);
}
}
