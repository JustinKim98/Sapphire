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
    auto& model = ModelManager::GetCurrentModel();

    auto& xDesc = model.GetDescriptor(xTensor.TensorDescriptorKey());
    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    Shape outputShape = xTensor.GetShape();
    const auto yKey = model.RegisterTensorDescriptor(outputShape, x.GetType(),
        x.GetCudaDevice());
    auto& yDesc = model.GetDescriptor(yKey);
    auto dy = yDesc.GetBackwardData();

    std::cout << "Basic Forward called" << std::endl;

    auto backPropWrapper =
        Util::SharedPtr<BackProp::BasicBackward>::Make(dx, dy);
    SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                std::make_tuple(&yDesc));

    return Tensor(yKey);
}

Tensor TwoInputs::operator()(Tensor& x1Tensor, Tensor& x2Tensor)
{
    auto& model = ModelManager::GetCurrentModel();

    auto& x1Desc = model.GetDescriptor(x1Tensor.TensorDescriptorKey());
    auto& x2Desc = model.GetDescriptor(x2Tensor.TensorDescriptorKey());
    auto x1 = x1Desc.GetForwardData();
    auto x2 = x2Desc.GetForwardData();
    auto dx1 = x1Desc.GetBackwardData();
    auto dx2 = x2Desc.GetBackwardData();
    Shape outputShape = x1Tensor.GetShape();
    const auto yKey = model.RegisterTensorDescriptor(outputShape, x1.GetType(),
        x1.GetCudaDevice());
    auto& yDesc = model.GetDescriptor(yKey);
    auto dy = yDesc.GetBackwardData();

    std::cout << "TwoInputs Forward called" << std::endl;

    auto backPropWrapper =
        Util::SharedPtr<BackProp::BackwardTwoInputs>::Make(dx1, dx2, dy);
    SaveHistory(backPropWrapper, std::make_tuple(&x1Desc, &x2Desc),
                std::make_tuple(&yDesc));

    return Tensor(yKey);
}

std::pair<Tensor, Tensor> TwoOutputs::operator()(Tensor& xTensor)
{
    auto& model = ModelManager::GetCurrentModel();

    auto& xDesc = model.GetDescriptor(xTensor.TensorDescriptorKey());
    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    Shape outputShape = xTensor.GetShape();
    const auto y1Key = model.RegisterTensorDescriptor(outputShape, x.GetType(),
        x.GetCudaDevice());
    const auto y2Key = model.RegisterTensorDescriptor(outputShape, x.GetType(),
        x.GetCudaDevice());
    auto& y1Desc = model.GetDescriptor(y1Key);
    auto& y2Desc = model.GetDescriptor(y2Key);
    auto dy1 = y1Desc.GetBackwardData();
    auto dy2 = y2Desc.GetBackwardData();

    std::cout << "TwoOutputs Forward called" << std::endl;

    auto backPropWrapper =
        Util::SharedPtr<BackProp::BackwardTwoOutputs>::Make(dx, dy1, dy2);
    SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                std::make_tuple(&y1Desc, &y2Desc));

    return std::make_pair(Tensor(y1Key), Tensor(y2Key));
}

void InplaceOp::operator()(Tensor& xTensor)
{
    auto& model = ModelManager::GetCurrentModel();

    auto& xDesc = model.GetDescriptor(xTensor.TensorDescriptorKey());
    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    Shape outputShape = xTensor.GetShape();

    std::cout << "In-place Forward called " << std::endl;

    auto backPropWrapper =
        Util::SharedPtr<BackProp::BackwardInplace>::Make(dx, dx);
    Util::SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&xDesc));
}
}
