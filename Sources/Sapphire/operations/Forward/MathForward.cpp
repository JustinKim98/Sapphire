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
Tensor MulOp(const Tensor& inputA, const Tensor& inputB)
{
    Model& model = ModelManager::GetCurrentModel();

    if (inputA.Mode() != inputB.Mode())
        throw std::invalid_argument("NN::Functional::MulOp - Mode mismatch");

    if (inputA.GetDevice() != inputB.GetDevice())
        throw std::invalid_argument("NN::Functional::MulOp - Device mismatch");

    auto mode = inputA.Mode();

    auto& aDesc =
        model.GetDescriptor(inputA.TensorDescriptorKey());
    auto& bDesc =
        model.GetDescriptor(inputB.TensorDescriptorKey());

    const auto aShape = aDesc.GetShape();
    const auto bShape = bDesc.GetShape();

    if (aShape.Cols() != bShape.Rows())
        throw std::invalid_argument("NN::Functional::MulOp - Shape mismatch");

    const auto yShapeOption =
        Util::GetBroadcastedShape(aDesc.GetShape(), bDesc.GetShape(), 2);
    if (!yShapeOption)
    {
        throw std::invalid_argument(
            "NN::Functional::MulOp - Broadcast failed");
    }

    auto yShape = yShapeOption.value();
    yShape.SetRow(inputA.GetShape().Rows());
    yShape.SetCol(inputB.GetShape().Cols());

    const Type type = aDesc.GetType();
    const CudaDevice device = aDesc.GetDevice();
    const int outputKey = model.RegisterTensorDescriptor(
        yShape, type, device);

    auto& yDesc = model.GetDescriptor(outputKey);
    yDesc.SetMode(mode);

    auto a = aDesc.GetForwardData();
    auto aCopy = a.CreateCopy();
    auto da = aDesc.GetBackwardData();
    auto b = bDesc.GetForwardData();
    auto bCopy = b.CreateCopy();
    auto db = bDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();

    Compute::Gemm(y, aCopy, bCopy, y);

    const auto backPropWrapper =
        Util::SharedPtr<BackProp::MulBackProp>::Make(aCopy, da, bCopy, db, y);

    Util::SaveHistory(backPropWrapper, std::make_tuple(&aDesc, &bDesc),
                      std::make_tuple(&yDesc));

    return Tensor(outputKey);
}

Tensor AddOp(const Tensor& inputA, const Tensor& inputB)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Get descriptors
    TensorUtil::TensorDescriptor& aDesc =
        model.GetDescriptor(inputA.TensorDescriptorKey());
    TensorUtil::TensorDescriptor& bDesc =
        model.GetDescriptor(inputB.TensorDescriptorKey());

    const auto shapeA = aDesc.GetShape();
    const auto shapeB = bDesc.GetShape();

    const auto outputShape = Util::GetBroadcastedShape(shapeA, shapeB, 0);
    if (!outputShape)
        throw std::invalid_argument("NN::Functional::AddOp - Broadcast failed");

    const Type type = aDesc.GetForwardData().GetType();
    const CudaDevice device = aDesc.GetForwardData().GetCudaDevice();

    const auto outKey = model.RegisterTensorDescriptor(
        outputShape.value(), type,
        device);
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
