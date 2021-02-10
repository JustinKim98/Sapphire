// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/compute/cuda/CudaParams.cuh>
#include <Motutapu/compute/cuda/dense/Basic.cuh>
#include <Motutapu/compute/cuda/dense/Gemm.cuh>
#include <Motutapu/compute/naive/NaiveBasic.hpp>
#include <Motutapu/compute/naive/NaiveGemm.hpp>
#include <algorithm>
#include <iostream>

namespace Motutapu::Compute
{
void Add(TensorData& out, const TensorData& a, const TensorData& b)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto broadcastA = a.BatchSize == 1;
    const auto broadcastB = b.BatchSize == 1;

    if ((out.TensorShape == a.TensorShape && out.TensorShape == b.TensorShape))
    {
        if (device.Type() == DeviceType::CUDA)
        {
            Cuda::Dense::Add(out.TensorShape.Size() * out.BatchSize,
                             out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             out.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::CPU)
        {
            Naive::Dense::Add(
                (out.TensorShape.Size() / N) * paddedN * out.BatchSize,
                out.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                (out.TensorShape.Size() / N) * paddedN, broadcastA, broadcastB);
            return;
        }
    }

    auto shapeOut = out.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    shapeOut.Expand(out.TensorShape.Dim() + 1);
    shapeA.Expand(out.TensorShape.Dim() + 1);
    shapeB.Expand(out.TensorShape.Dim() + 1);

    shapeOut.Set(0, out.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             0, 0, Cuda::Dense::Add, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, out.DenseMatHost,
                             a.DenseMatHost, b.DenseMatHost, 0, 1,
                             Naive::Dense::Add, 0, false, false);
    }
}

void Sub(TensorData& out, const TensorData& a, const TensorData& b)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto broadcastA = a.BatchSize == 1;
    const auto broadcastB = b.BatchSize == 1;

    if ((out.TensorShape == a.TensorShape && out.TensorShape == b.TensorShape))
    {
        if (device.Type() == DeviceType::CUDA)
        {
            Cuda::Dense::Sub(out.TensorShape.Size() * out.BatchSize,
                             out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             out.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::CPU)
        {
            Naive::Dense::Sub(
                (out.TensorShape.Size() / N) * paddedN * out.BatchSize,
                out.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                (out.TensorShape.Size() / N) * paddedN, broadcastA, broadcastB);
            return;
        }
    }

    auto shapeOut = out.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    shapeOut.Expand(out.TensorShape.Dim() + 1);
    shapeA.Expand(out.TensorShape.Dim() + 1);
    shapeB.Expand(out.TensorShape.Dim() + 1);

    shapeOut.Set(0, out.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             0, 0, Cuda::Dense::Sub, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, out.DenseMatHost,
                             a.DenseMatHost, b.DenseMatHost, 0, 1,
                             Naive::Dense::Sub, 0, false, false);
    }
}

void Dot(TensorData& out, const TensorData& a, const TensorData& b)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto broadcastA = a.BatchSize == 1;
    const auto broadcastB = b.BatchSize == 1;

    if ((out.TensorShape == a.TensorShape && out.TensorShape == b.TensorShape))
    {
        if (device.Type() == DeviceType::CUDA)
        {
            Cuda::Dense::Dot(out.TensorShape.Size() * out.BatchSize,
                             out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             out.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::CPU)
        {
            Naive::Dense::Dot(
                (out.TensorShape.Size() / N) * paddedN * out.BatchSize,
                out.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                (out.TensorShape.Size() / N) * paddedN, broadcastA, broadcastB);
            return;
        }
    }

    auto shapeOut = out.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    shapeOut.Expand(out.TensorShape.Dim() + 1);
    shapeA.Expand(out.TensorShape.Dim() + 1);
    shapeB.Expand(out.TensorShape.Dim() + 1);

    shapeOut.Set(0, out.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             0, 0, Cuda::Dense::Dot, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, out.DenseMatHost,
                             a.DenseMatHost, b.DenseMatHost, 0, 1,
                             Naive::Dense::Dot, 0, false, false);
    }
}

void Gemm(TensorUtil::TensorData& out, const TensorUtil::TensorData& a,
          const TensorUtil::TensorData& b, const TensorUtil::TensorData& c)
{
    const auto device = out.GetDevice();
    const auto M = out.Rows();
    const auto N = out.Cols();
    const auto K = a.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto paddedK = a.PaddedHostColSize;
    const auto batchSize = out.BatchSize;

    //! Faster broadcast multiply for Cuda if all tensor dimensions are fixed to
    //! 2
    if (out.TensorShape.Dim() == 2 && a.TensorShape.Dim() == 2 &&
        b.TensorShape.Dim() == 2 && c.TensorShape.Dim() == 2)
    {
        if (device.Type() == DeviceType::CUDA)
        {
            Cuda::Dense::GemmMatrixWiseBroadcast(
                out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                c.DenseMatCuda, M, N, K, batchSize, a.BatchSize == 1,
                b.BatchSize == 1, c.BatchSize == 1);
            return;
        }
        else
        {
        }
    }

    //! Treat batch size as part of tensor shape
    auto shapeOut = out.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;
    auto shapeC = c.TensorShape;

    shapeOut.Expand(out.TensorShape.Dim() + 1);
    shapeA.Expand(out.TensorShape.Dim() + 1);
    shapeB.Expand(out.TensorShape.Dim() + 1);
    shapeC.Expand(out.TensorShape.Dim() + 1);

    shapeOut.Set(0, out.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);
    shapeC.Set(0, c.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();
    const auto sizeC = shapeC.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        cublasHandle_t cublasHandle;
        cublasCreate(&cublasHandle);

        BroadcastWith3Inputs(shapeOut, shapeA, shapeB, shapeC, sizeOut, sizeA,
                             sizeB, sizeC, out.DenseMatCuda, a.DenseMatCuda,
                             b.DenseMatCuda, c.DenseMatCuda, 0, 2,
                             Cuda::Dense::Gemm, M, N, K, &cublasHandle);

        cublasDestroy(cublasHandle);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / K) * paddedK;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        const auto paddedSizeC = (sizeC / N) * paddedN;

        BroadcastWith3Inputs(shapeOut, shapeA, shapeB, shapeC, paddedSizeOut,
                             paddedSizeA, paddedSizeB, paddedSizeC,
                             out.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                             c.DenseMatHost, 0, 2, Naive::Dense::NaiveGemm, M,
                             N, paddedN, K, paddedK);
    }
}

void Scale(TensorData& output, const TensorData& input, const float factor)
{
    const auto device = output.GetDevice();
    const auto N = output.Cols();
    const auto paddedN = output.PaddedHostColSize;
    const auto totalSize = output.TensorShape.Size() * output.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Scale(output.DenseMatCuda, input.DenseMatCuda, factor,
                           totalSize);
    }
    else
    {
        Naive::Dense::Scale(output.DenseMatHost, input.DenseMatHost, factor,
                            totalSizeWithPadding);
    }
}

void Transpose(TensorData& output, const TensorData& input)
{
    const auto device = output.GetDevice();
    const auto M = output.Rows();
    const auto N = output.Cols();
    const auto paddedN = output.PaddedHostColSize;
    const auto broadcast = input.BatchSize == 1;

    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Transpose(output.DenseMatCuda, input.DenseMatCuda, M, N,
                               output.BatchSize, broadcast);
    }
    else
    {
        Naive::Dense::Transpose(output.DenseMatHost, input.DenseMatHost, M,
                                paddedN, output.BatchSize, broadcast);
    }
}

}  // namespace Motutapu::Compute