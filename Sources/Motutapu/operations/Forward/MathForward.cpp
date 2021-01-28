// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Model.hpp>
#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/operations/Backward/MathBackward.hpp>
#include <Motutapu/operations/Forward/MathForward.hpp>
#include <vector>

namespace Motutapu::Functional
{
__attribute__((unused))
static Tensor MulOp(const Tensor& a, const Tensor& b)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Perform out = a*b
    Util::TensorDescriptor& descA =
        model.GetDescriptor(a.TensorDescriptorKey());
    Util::TensorDescriptor& descB =
        model.GetDescriptor(b.TensorDescriptorKey());

    auto shapeA = descA.ForwardData.TensorShape;
    auto shapeB = descB.ForwardData.TensorShape;

    const auto batchSize = descA.ForwardData.BatchSize;
    Type type = descA.ForwardData.GetType();
    Device device = descA.ForwardData.GetDevice();

    const auto outputShape = Shape({ shapeA.At(0), shapeB.At(1) });

    Util::TensorDescriptor descOut(outputShape, type, device, batchSize, false);
    model.RegisterTensorDescriptor(descOut);

    Compute::Mul(descOut.ForwardData, descA.ForwardData, descB.ForwardData);

    auto backPropWrapper =
        std::make_unique<BackProp::MulBackProp>(descA.Key, descB.Key);

    descA.AppendOperandHistory(descOut.Key);
    descB.AppendOperandHistory(descOut.Key);
    descOut.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(outputShape, descOut.Key);
}

__attribute__((unused))
static Tensor AddOp(const Tensor& a, const Tensor& b)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Perform out = a*b
    Util::TensorDescriptor& descA =
        model.GetDescriptor(a.TensorDescriptorKey());
    Util::TensorDescriptor& descB =
        model.GetDescriptor(b.TensorDescriptorKey());

    auto shapeA = descA.ForwardData.TensorShape;
    auto shapeB = descB.ForwardData.TensorShape;

    const auto batchSize = descA.ForwardData.BatchSize;
    Type type = descA.ForwardData.GetType();
    Device device = descA.ForwardData.GetDevice();

    const auto outputShape = Shape({ shapeA.At(0), shapeA.At(1) });

    Util::TensorDescriptor descOut(outputShape, type, device, batchSize, false);
    model.RegisterTensorDescriptor(descOut);

    Compute::Add(descOut.ForwardData, descA.ForwardData, descB.ForwardData);

    auto backPropWrapper =
        std::make_unique<BackProp::AddBackProp>(descA.Key, descB.Key);

    descA.AppendOperandHistory(descOut.Key);
    descB.AppendOperandHistory(descOut.Key);
    descOut.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(outputShape, descOut.Key);
}

__attribute__((unused))
static void AddOpInplace(const Tensor& out, Tensor& a)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Perform out = a*b
    Util::TensorDescriptor& descA =
        model.GetDescriptor(a.TensorDescriptorKey());
    Util::TensorDescriptor& descOut =
        model.GetDescriptor(out.TensorDescriptorKey());

    auto shapeA = descA.ForwardData.TensorShape;
    const auto outputShape = descOut.ForwardData.TensorShape;


    const auto batchSize = descA.ForwardData.BatchSize;

    Compute::Add(descOut.ForwardData, descA.ForwardData);

    auto backPropWrapper =
        std::make_unique<BackProp::AddBackPropInplace>(descA.Key);

    descA.AppendOperandHistory(descOut.Key);
    descOut.AppendOperandHistory(descOut.Key);
    descOut.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(outputShape, descOut.Key);
}
}  // namespace Motutapu::Functional
