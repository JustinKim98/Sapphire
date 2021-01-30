// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Model.hpp>
#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/operations/Backward/MathBackward.hpp>
#include <Motutapu/operations/Forward/MathForward.hpp>
#include <vector>

namespace Motutapu::NN::Functional
{
static Tensor MulOp(const Tensor& a, const Tensor& b)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Perform out = a*b
    TensorUtil::TensorDescriptor& descA =
        model.GetDescriptor(a.TensorDescriptorKey());
    TensorUtil::TensorDescriptor& descB =
        model.GetDescriptor(b.TensorDescriptorKey());

    auto shapeA = descA.ForwardData.TensorShape;
    auto shapeB = descB.ForwardData.TensorShape;

    const auto batchSize = descA.ForwardData.BatchSize;
    Type type = descA.ForwardData.GetType();
    Device device = descA.ForwardData.GetDevice();

    const auto outputShape = Shape({ shapeA.At(0), shapeB.At(1) });

    TensorUtil::TensorDescriptor descOut(outputShape, type, device, batchSize, false);
    const auto outputKey = model.RegisterTensorDescriptor(descOut);

    Compute::Mul(descOut.ForwardData, descA.ForwardData, descB.ForwardData);

    auto backPropWrapper =
        std::make_unique<BackProp::MulBackProp>(descA.Key, descB.Key);

    descA.AppendOperandHistory(descOut.Key);
    descB.AppendOperandHistory(descOut.Key);
    descOut.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(outputShape, outputKey);
}

static Tensor AddOp(const Tensor& a, const Tensor& b)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Get descriptors
    TensorUtil::TensorDescriptor& descA =
        model.GetDescriptor(a.TensorDescriptorKey());
    TensorUtil::TensorDescriptor& descB =
        model.GetDescriptor(b.TensorDescriptorKey());

    auto shapeA = descA.ForwardData.TensorShape;
    auto shapeB = descB.ForwardData.TensorShape;

    const auto batchSize = descA.ForwardData.BatchSize;
    Type type = descA.ForwardData.GetType();
    Device device = descA.ForwardData.GetDevice();

    const auto outputShape = Shape({ shapeA.At(0), shapeA.At(1) });

    TensorUtil::TensorDescriptor descOut(outputShape, type, device, batchSize, false);
    model.RegisterTensorDescriptor(descOut);

    Compute::Add(descOut.ForwardData, descA.ForwardData, descB.ForwardData);

    auto backPropWrapper =
        std::make_unique<BackProp::AddBackProp>(descA.Key, descB.Key);

    descA.AppendOperandHistory(descOut.Key);
    descB.AppendOperandHistory(descOut.Key);
    descOut.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(outputShape, descOut.Key);
}

static void AddOpInplace(const Tensor& out, Tensor& a)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Get descriptors
    TensorUtil::TensorDescriptor& descA =
        model.GetDescriptor(a.TensorDescriptorKey());
    TensorUtil::TensorDescriptor& descOut =
        model.GetDescriptor(out.TensorDescriptorKey());

    //! Derive output shape
    auto shapeA = descA.ForwardData.TensorShape;
    const auto outputShape = descOut.ForwardData.TensorShape;

    Compute::Add(descOut.ForwardData, descA.ForwardData);

    auto backPropWrapper =
        std::make_unique<BackProp::AddBackPropInplace>(descA.Key);

    descA.AppendOperandHistory(descOut.Key);
    descOut.AppendOperandHistory(descOut.Key);
    descOut.AppendOutputHistory(std::move(backPropWrapper), false);
}
}  // namespace Motutapu::Functional
