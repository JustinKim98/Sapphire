// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Forward/Basic.hpp>
#include <Sapphire/operations/Backward/BasicBackward.hpp>
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/Model.hpp>
#include <iostream>

namespace Sapphire::NN
{
Tensor Basic::operator()(Tensor& xTensor)
{
    auto& model = ModelManager::CurModel();

    auto& xDesc = model.GetDescriptor(xTensor.TensorDescriptorKey());
    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    Shape outputShape = xTensor.GetShape();
    const auto yKey = model.RegisterTensorDescriptor(
        outputShape, xDesc.GetType(),
        xDesc.GetDevice());
    auto& yDesc = model.GetDescriptor(yKey);
    auto dy = yDesc.GetBackwardData();

    std::cout << "Basic Forward called" << std::endl;

    auto* wrapper = new BackProp::BasicBackward(dx, dy);
    Util::SaveHistory(wrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&yDesc));

    return Tensor(yKey);
}

Tensor TwoInputs::operator()(Tensor& x1Tensor, Tensor& x2Tensor)
{
    auto& model = ModelManager::CurModel();

    auto& x1Desc = model.GetDescriptor(x1Tensor.TensorDescriptorKey());
    auto& x2Desc = model.GetDescriptor(x2Tensor.TensorDescriptorKey());
    auto x1 = x1Desc.GetForwardData();
    auto x2 = x2Desc.GetForwardData();
    auto dx1 = x1Desc.GetBackwardData();
    auto dx2 = x2Desc.GetBackwardData();
    Shape outputShape = x1Tensor.GetShape();
    const auto yKey = model.RegisterTensorDescriptor(
        outputShape, x1Desc.GetType(),
        x1Desc.GetDevice());
    auto& yDesc = model.GetDescriptor(yKey);
    auto dy = yDesc.GetBackwardData();

    std::cout << "TwoInputs Forward called" << std::endl;

    auto* wrapper = new BackProp::BackwardTwoInputs(dx1, dx2, dy);
    Util::SaveHistory(wrapper, std::make_tuple(&x1Desc, &x2Desc),
                      std::make_tuple(&yDesc));

    return Tensor(yKey);
}

std::pair<Tensor, Tensor> TwoOutputs::operator()(Tensor& xTensor)
{
    auto& model = ModelManager::CurModel();

    auto& xDesc = model.GetDescriptor(xTensor.TensorDescriptorKey());
    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    Shape outputShape = xTensor.GetShape();
    const auto y1Key = model.RegisterTensorDescriptor(
        outputShape, xDesc.GetType(),
        xDesc.GetDevice());
    const auto y2Key = model.RegisterTensorDescriptor(
        outputShape, xDesc.GetType(),
        xDesc.GetDevice());
    auto& y1Desc = model.GetDescriptor(y1Key);
    auto& y2Desc = model.GetDescriptor(y2Key);
    auto dy1 = y1Desc.GetBackwardData();
    auto dy2 = y2Desc.GetBackwardData();

    std::cout << "TwoOutputs Forward called" << std::endl;

    auto* wrapper = new BackProp::BackwardTwoOutputs(dx, dy1, dy2);
    Util::SaveHistory(std::move(wrapper), std::make_tuple(&xDesc),
                      std::make_tuple(&y1Desc, &y2Desc));

    return std::make_pair(Tensor(y1Key), Tensor(y2Key));
}

void InplaceOp::operator()(Tensor& xTensor)
{
    auto& model = ModelManager::CurModel();

    auto& xDesc = model.GetDescriptor(xTensor.TensorDescriptorKey());
    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    Shape outputShape = xTensor.GetShape();

    std::cout << "In-place Forward called " << std::endl;

    auto* wrapper = new BackProp::BackwardInplace(dx, dx);
    Util::SaveHistory(std::move(wrapper), std::make_tuple(&xDesc),
                      std::make_tuple(&xDesc));
}
}
