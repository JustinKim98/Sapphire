// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_FUNCTIONAL_MATH_DECL_HPP
#define MOTUTAPU_FUNCTIONAL_MATH_DECL_HPP

#include <Motutapu/Model.hpp>
#include <Motutapu/operations/Backward/MathBackwardDecl.hpp>
#include <Motutapu/compute/Compute.hpp>

namespace Motutapu::Functional
{
template <typename T>
static Tensor<T> MulOp(const Tensor<T>& a, const Tensor<T>& b)
{
    Model& model = ModelManager::GetCurrentModel();

    //! Perform out = a*b
    Util::TensorDescriptor<T>& descA =
        model.GetDescriptor(a.TensorDescriptorKey());
    Util::TensorDescriptor<T>& descB =
        model.GetDescriptor(b.TensorDescriptorKey());

    auto shapeA = descA.ForwardData.TensorShape;
    auto shapeB = descB.ForwardData.TensorShape;

    const auto batchSize = descA.BatchSize;
    Type type = descA.ForwardData.GetType();
    Device device = descA.ForwardData.GetDevice();

    const auto outputShape = Shape({ shapeA.At(0), shapeB.At(1) });

    Util::TensorDescriptor<T> descOut(outputShape, type, device, batchSize);
    model.RegisterTensorDescriptor<T>(descOut);

    Compute::Mul<T>(descOut->ForwardData, descA->ForwardData,
                    descB->ForwardData);

    auto backPropWrapper = BackProp::MulBackProp<T>(descA.Key, descB.Key);

    descA.AppendOperandHistory(descOut.Key);
    descB.AppendOperandHistory(descOut.Key);
    descOut.AppendOutputUnitHistory(backPropWrapper, false);

    return Tensor<T>(outputShape, descOut);
}
}

#endif
