// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Forward/ReLU.hpp>
#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/operations/Backward/ReLUBackward.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/compute/ActivationOps.hpp>

namespace Sapphire::NN
{
Tensor ReLU(Tensor xTensor)
{
    Model& model = ModelManager::GetCurrentModel();
    auto& xDesc = model.GetDescriptor(xTensor.TensorDescriptorKey());
    const auto yDescKey = model.RegisterTensorDescriptor(
        xDesc.GetShape(), xDesc.GetType(), xDesc.GetDevice());
    auto& yDesc = model.GetDescriptor(yDescKey);
    yDesc.SetMode(xDesc.Mode());

    auto x = xDesc.GetForwardData().CreateCopy();
    auto dx = xDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();
    auto wrapper = Util::SharedPtr<BackProp::ReLUBackward>::Make(dx, dy, x);
    Util::SaveHistory(wrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&yDesc));
    Util::ChangeTensorDataDimension(1, x, dx, y, dy);
    Compute::ReLU(y, x);

    return Tensor(yDescKey);
}
}
