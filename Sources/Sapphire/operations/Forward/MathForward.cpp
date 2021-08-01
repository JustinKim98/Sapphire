// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/MathBackward.hpp>
#include <Sapphire/operations/Forward/MathForward.hpp>
#include <Sapphire/util/SharedPtr.hpp>
#include <Sapphire/util/UnitUtils.hpp>

namespace Sapphire::NN::Functional
{
static Tensor MulOp(const Tensor& inputA, const Tensor& inputB)
{
    Model& model = ModelManager::GetCurrentModel();

    auto& aDesc =
        model.GetDescriptor(inputA.TensorDescriptorKey());
    auto& bDesc =
        model.GetDescriptor(inputB.TensorDescriptorKey());

    if (Util::CheckBatchSizeEquality(aDesc, bDesc))
        throw std::invalid_argument(
            "NN::Functional::MulOp - Given tensors have different batch size");
    if (Util::CheckDeviceEquality(aDesc, bDesc))
        throw std::invalid_argument(
            "NN::Functional::MulOp - Given tensors are not on the same device");

    const auto outputShape =
        Util::GetBroadcastedShape(aDesc.GetShape(), bDesc.GetShape());
    if (!outputShape)
    {
        throw std::invalid_argument(
            "NN::Functional::MulOp - Given shapes failed to broadcast");
    }

    const auto batchSize = aDesc.GetBatchSize();
    const Type type = aDesc.GetType();
    const Device device = aDesc.GetDevice();
    const int outputKey = model.RegisterTensorDescriptor(
        outputShape.value(), type, device, batchSize);

    auto& yDesc = model.GetDescriptor(outputKey);

    auto a = aDesc.GetForwardData().CreateCopy();
    auto da = aDesc.GetBackwardData();
    auto b = bDesc.GetForwardData().CreateCopy();
    auto db = bDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();

    const auto backPropWrapper =
        Util::SharedPtr<BackProp::MulBackProp>::Make(a, da, b, db, y);

    Util::SaveHistory(backPropWrapper, std::make_tuple(&aDesc, &bDesc),
                      std::make_tuple(&yDesc));

    Compute::Gemm(y, a, b, y);
    return Tensor(outputKey);
}

static Tensor AddOp(const Tensor& inputA, const Tensor& inputB)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Get descriptors
    TensorUtil::TensorDescriptor& aDesc =
        model.GetDescriptor(inputA.TensorDescriptorKey());
    TensorUtil::TensorDescriptor& bDesc =
        model.GetDescriptor(inputB.TensorDescriptorKey());

    const auto shapeA = aDesc.GetShape();
    const auto shapeB = bDesc.GetShape();

    const auto batchSize = aDesc.GetBatchSize();
    const Type type = aDesc.GetForwardData().GetType();
    const Device device = aDesc.GetForwardData().GetDevice();

    const auto outputShape = Shape({ shapeA.At(0), shapeA.At(1) });

    const auto outKey = model.RegisterTensorDescriptor(outputShape, type,
        device, batchSize);
    auto& yDesc = model.GetDescriptor(outKey);

    auto a = aDesc.GetForwardData().CreateCopy();
    auto da = aDesc.GetBackwardData();
    auto b = bDesc.GetForwardData().CreateCopy();
    auto db = bDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();

    const auto backPropWrapper =
        Util::SharedPtr<BackProp::AddBackProp>::Make(da, db, dy);
    Util::SaveHistory(backPropWrapper, std::make_tuple(&aDesc, &bDesc),
                      std::make_tuple(&yDesc));

    Compute::Add(y, a, b);
    return Tensor(yDesc.GetKey());
}
} // namespace Sapphire::NN::Functional
