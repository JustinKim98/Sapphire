// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/operations/Forward/MathForward.hpp>
#include <Motutapu/operations/Backward/MathBackward.hpp>
#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/Model.hpp>
#include <vector>

namespace Motutapu::Functional
{
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

    Util::TensorDescriptor descOut(outputShape, type, device, batchSize);
    model.RegisterTensorDescriptor(descOut);

    Compute::Mul(descOut.ForwardData, descA.ForwardData, descB.ForwardData);

    auto backPropWrapper = std::make_unique<BackProp::MulBackProp>(
        std::vector({ descA.Key, descB.Key }));

    descA.AppendOperandHistory(descOut.Key);
    descB.AppendOperandHistory(descOut.Key);
    descOut.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(outputShape, descOut.Key);
}
} // namespace Motutapu::Functional
