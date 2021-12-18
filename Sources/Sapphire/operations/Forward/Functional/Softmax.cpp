// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Forward/Functional/Softmax.hpp>
#include <Sapphire/compute/ActivationOps.hpp>
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/operations/Backward/SoftmaxBackward.hpp>
#include <Sapphire/Model.hpp>

namespace Sapphire::F
{
Tensor SoftMax(const Tensor& input)
{
    static int unitIdCount = 0;
    Model& model = ModelManager::CurModel();
    auto& xDesc = model.GetDescriptor(input.TensorDescriptorKey());
    const auto yDescKey = model.RegisterTensorDescriptor(
        input.GetShape(), xDesc.GetType(),
        xDesc.GetDevice());
    auto& yDesc = model.GetDescriptor(yDescKey);

    auto x = xDesc.GetForwardData().CreateCopy();
    auto dx = xDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();
    Util::ChangeTensorDataDimension(2, x, dx, y, dy);
    Compute::SoftMax(y, x);
    auto* wrapper = new BackProp::SoftMaxBackward(
        "Softmax" + std::to_string(unitIdCount++), dx, dy, y);
    Util::SaveHistory(wrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&yDesc));

    return Tensor(yDescKey);
}
} // namespace Sapphire::NN
