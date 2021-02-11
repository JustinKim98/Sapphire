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
static Tensor Mul(const Tensor& a, const Tensor& b)
{
    Model& model = ModelManager::GetCurrentModel();

    TensorUtil::TensorDescriptor& descA =
        model.GetDescriptor(a.TensorDescriptorKey());
    TensorUtil::TensorDescriptor& descB =
        model.GetDescriptor(b.TensorDescriptorKey());

    Shape shapeA = descA.ForwardData.TensorShape;
    Shape shapeB = descB.ForwardData.TensorShape;

    const auto batchSize = descA.ForwardData.BatchSize;
    Type type = descA.ForwardData.GetType();
    Device device = descA.ForwardData.GetDevice();

    const Shape outputShape({ shapeA.At(0), shapeB.At(1) });

    const int outputKey =
        model.RegisterTensorDescriptor(outputShape, type, device, batchSize);

    auto& descOut = model.GetDescriptor(outputKey);

    Compute::Gemm(descOut.ForwardData, descA.ForwardData, descB.ForwardData,
                  descOut.ForwardData);

    auto backPropWrapper = std::make_unique<BackProp::MulBackProp>(
        descA.ForwardData, descA.BackwardData, descB.ForwardData,
        descB.BackwardData, descOut.BackwardData);

    //! Append operand history to the descriptors of A and B
    descA.AppendOperandHistory(descOut.GetKey());
    descB.AppendOperandHistory(descOut.GetKey());
    //! Append output history to the descriptor A and associated backPropWrapper
    descOut.AppendOutputHistory(std::move(backPropWrapper), false);

    return Tensor(outputShape, outputKey);
}
//
// static Tensor AddOp(const Tensor& a, const Tensor& b)
//{
//    Model& model = ModelManager::GetCurrentModel();
//
//    //! Get descriptors
//    TensorUtil::TensorDescriptor& descA =
//        model.GetDescriptor(a.TensorDescriptorKey());
//    TensorUtil::TensorDescriptor& descB =
//        model.GetDescriptor(b.TensorDescriptorKey());
//
//    auto shapeA = descA.ForwardData.TensorShape;
//    auto shapeB = descB.ForwardData.TensorShape;
//
//    const auto batchSize = descA.ForwardData.BatchSize;
//    Type type = descA.ForwardData.GetType();
//    Device device = descA.ForwardData.GetDevice();
//
//    const auto outputShape = Shape({ shapeA.At(0), shapeA.At(1) });
//
//    TensorUtil::TensorDescriptor descOut(outputShape, type, device, batchSize,
//                                         false);
//    model.RegisterTensorDescriptor(descOut);
//
//    Compute::Add(descOut.ForwardData, descA.ForwardData, descB.ForwardData);
//
//    auto backPropWrapper =
//        std::make_unique<BackProp::AddBackProp>(descA.m_key, descB.m_key);
//
//    descA.AppendOperandHistory(descOut.m_key);
//    descB.AppendOperandHistory(descOut.m_key);
//    descOut.AppendOutputHistory(std::move(backPropWrapper), false);
//
//    return Tensor(outputShape, descOut.m_key);
//}
//
// static void AddOpInplace(const Tensor& out, Tensor& a)
//{
//    Model& model = ModelManager::GetCurrentModel();
//
//    //! Get descriptors
//    TensorUtil::TensorDescriptor& descA =
//        model.GetDescriptor(a.TensorDescriptorKey());
//    TensorUtil::TensorDescriptor& descOut =
//        model.GetDescriptor(out.TensorDescriptorKey());
//
//    //! Derive output shape
//    auto shapeA = descA.ForwardData.TensorShape;
//    const auto outputShape = descOut.ForwardData.TensorShape;
//
//    Compute::Add(descOut.ForwardData, descA.ForwardData);
//
//    auto backPropWrapper =
//        std::make_unique<BackProp::AddBackPropInplace>(descA.m_key);
//
//    descA.AppendOperandHistory(descOut.m_key);
//    descOut.AppendOperandHistory(descOut.m_key);
//    descOut.AppendOutputHistory(std::move(backPropWrapper), false);
//}
}  // namespace Motutapu::NN::Functional
