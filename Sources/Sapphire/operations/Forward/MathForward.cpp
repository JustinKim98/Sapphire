// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/MathBackward.hpp>
#include <Sapphire/operations/Forward/MathForward.hpp>

namespace Sapphire::NN::Functional
{
static Tensor MulOp(const Tensor& a, const Tensor& b)
{
    Model& model = ModelManager::GetCurrentModel();

    TensorUtil::TensorDescriptor& aDesc =
        model.GetDescriptor(a.TensorDescriptorKey());
    TensorUtil::TensorDescriptor& bDesc =
        model.GetDescriptor(b.TensorDescriptorKey());

    const Shape shapeA = aDesc.ForwardData.TensorShape;
    const Shape shapeB = bDesc.ForwardData.TensorShape;

    const auto batchSize = aDesc.ForwardData.BatchSize;
    const Type type = aDesc.ForwardData.GetType();
    const Device device = aDesc.ForwardData.GetDevice();

    const Shape outputShape({ shapeA.At(0), shapeB.At(1) });

    const int outputKey = model.RegisterTensorDescriptor(
        outputShape, type, device, batchSize, true);

    auto& yDesc = model.GetDescriptor(outputKey);

    Compute::Gemm(yDesc.ForwardData, aDesc.ForwardData, bDesc.ForwardData,
                  yDesc.ForwardData);

    auto backPropWrapper = std::make_unique<BackProp::MulBackProp>(
        aDesc.ForwardData, aDesc.BackwardData, bDesc.ForwardData,
        bDesc.BackwardData, yDesc.BackwardData);

    //! Append operand history to the descriptors of A and B
    aDesc.AppendOperandHistory(yDesc.GetKey());
    bDesc.AppendOperandHistory(yDesc.GetKey());
    //! Append output history to the descriptor A and associated backPropWrapper
    yDesc.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(outputKey);
}

static Tensor AddOp(const Tensor& a, const Tensor& b)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Get descriptors
    TensorUtil::TensorDescriptor& descA =
        model.GetDescriptor(a.TensorDescriptorKey());
    TensorUtil::TensorDescriptor& descB =
        model.GetDescriptor(b.TensorDescriptorKey());

    const auto shapeA = descA.ForwardData.TensorShape;
    const auto shapeB = descB.ForwardData.TensorShape;

    const auto batchSize = descA.ForwardData.BatchSize;
    const Type type = descA.ForwardData.GetType();
    const Device device = descA.ForwardData.GetDevice();

    const auto outputShape = Shape({ shapeA.At(0), shapeA.At(1) });

    const auto outKey = model.RegisterTensorDescriptor(outputShape, type,
        device, batchSize, true);
    auto& descOut = model.GetDescriptor(outKey);

    Compute::Add(descOut.ForwardData, descA.ForwardData, descB.ForwardData);

    auto backPropWrapper = std::make_unique<BackProp::AddBackProp>(
        descA.BackwardData, descB.BackwardData, descOut.BackwardData);

    descA.AppendOperandHistory(descOut.GetKey());
    descB.AppendOperandHistory(descOut.GetKey());
    descOut.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(descOut.GetKey());
}
} // namespace Sapphire::NN::Functional
