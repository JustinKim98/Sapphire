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
// void Add(TensorData& out, const TensorData& add)
//{
//    const auto device = out.GetDevice();
//    const auto M = out.Rows();
//    const auto N = out.Cols();
//    const auto paddedN = out.PaddedHostColSize;
//    const auto stride = M * N;
//    const auto strideWithPadding = M * paddedN;
//    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
//    const auto totalSizeWithPadding = (totalSize / N) * paddedN;
//    const auto broadcast = add.BatchSize == 1;
//
//    if (device.Type() == DeviceType::CUDA)
//    {
//        Cuda::Dense::Add(out.DenseMatCuda, out.DenseMatCuda, add.DenseMatCuda,
//                         totalSize, stride, false, broadcast);
//    }
//    else
//    {
//        Naive::Dense::Add(out.DenseMatHost, out.DenseMatHost,
//        add.DenseMatHost,
//                          totalSizeWithPadding, strideWithPadding, false,
//                          broadcast);
//    }
//}

// void Add(TensorData& out, const TensorData& a, const TensorData& b)
//{
//    const auto device = out.GetDevice();
//    const auto M = out.Rows();
//    const auto N = out.Cols();
//    const auto paddedN = out.PaddedHostColSize;
//    const auto stride = M * N;
//    const auto strideWithPadding = M * paddedN;
//    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
//    const auto totalSizeWithPadding = (totalSize / N) * paddedN;
//    const auto broadcastA = a.BatchSize == 1;
//    const auto broadcastB = b.BatchSize == 1;
//
//    if (device.Type() == DeviceType::CUDA)
//    {
//        Cuda::Dense::Add(out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
//                         totalSize, stride, broadcastA, broadcastB);
//    }
//    else
//    {
//        Naive::Dense::Add(out.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
//                          totalSizeWithPadding, strideWithPadding, broadcastA,
//                          broadcastB);
//    }
//}

void Sub(TensorData& out, const TensorData& sub)
{
    const auto device = out.GetDevice();
    const auto M = out.Rows();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto stride = M * N;
    const auto strideWithPadding = M * paddedN;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;
    const auto broadcast = sub.BatchSize == 1;

    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Sub(out.DenseMatCuda, out.DenseMatCuda, sub.DenseMatCuda,
                         totalSize, stride, false, broadcast);
    }
    else
    {
        Naive::Dense::Sub(out.DenseMatHost, out.DenseMatHost, sub.DenseMatHost,
                          totalSizeWithPadding, strideWithPadding, false,
                          broadcast);
    }
}

void Sub(TensorData& out, const TensorData& a, const TensorData& b)
{
    const auto device = out.GetDevice();
    const auto M = out.Rows();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto stride = M * N;
    const auto strideWithPadding = M * paddedN;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;
    const auto broadcastA = a.BatchSize == 1;
    const auto broadcastB = b.BatchSize == 1;

    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Sub(out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                         totalSize, stride, broadcastA, broadcastB);
    }
    else
    {
        Naive::Dense::Sub(out.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                          totalSizeWithPadding, strideWithPadding, broadcastA,
                          broadcastB);
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
    const auto sizeOut = out.TensorShape.Size();
    const auto sizeA = a.TensorShape.Size();
    const auto sizeB = b.TensorShape.Size();
    const auto sizeC = c.TensorShape.Size();

    //    if (out.TensorShape.Dim() == 2 && a.TensorShape.Dim() == 2 &&
    //        b.TensorShape.Dim() == 2 && c.TensorShape.Dim() == 2)
    //    {
    //        if (device.Type() == DeviceType::CUDA)
    //        {
    //            Cuda::Dense::GemmMatrixWiseBroadcast(
    //                out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
    //                c.DenseMatCuda, M, N, K, batchSize, a.BatchSize == 1,
    //                b.BatchSize == 1, c.BatchSize == 1);
    //            return;
    //        }
    //        else
    //        {
    //        }
    //    }
    //! Unify batch with given tensor shape??
    //    auto shapeOut = out.TensorShape;
    //    auto shapeA = a.TensorShape;
    //    auto shapeB = b.TensorShape;
    //    auto shapeC = c.TensorShape;
    //
    //    shapeOut.Expand(sizeOut + 1);
    //    shapeA.Expand(sizeOut + 1);
    //    shapeB.Expand(sizeOut + 1);
    //    shapeC.Expand(sizeOut + 1);
    //
    //    shapeOut.Set(0, out.BatchSize);
    //    shapeA.Set(0, a.BatchSize);
    //    shapeB.Set(0, b.BatchSize);
    //    shapeC.Set(0, c.BatchSize);

    if (device.Type() == DeviceType::CUDA)
    {
        cublasHandle_t cublasHandle;
        cublasCreate(&cublasHandle);

        cudaStream_t streams[batchSize];
        for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            cudaStreamCreate(&streams[batchIdx]);

        for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            broadcastWith3Inputs(
                out.TensorShape, a.TensorShape, b.TensorShape, c.TensorShape,
                sizeOut, sizeA, sizeB, sizeC,
                out.DenseMatCuda + batchIdx * sizeOut,
                a.DenseMatCuda + (batchIdx % a.BatchSize) * sizeA,
                b.DenseMatCuda + (batchIdx % b.BatchSize) * sizeB,
                c.DenseMatCuda + (batchIdx % c.BatchSize) * sizeC, 0, 2,
                Cuda::Dense::Gemm, M, N, K, &cublasHandle, streams);
        }
        cudaDeviceSynchronize();
        for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            cudaStreamDestroy(streams[batchIdx]);
        cublasDestroy(cublasHandle);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / K) * paddedK;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        const auto paddedSizeC = (sizeC / N) * paddedN;

        //#pragma omp parallel for schedule(static)
        for (unsigned int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            broadcastWith3Inputs(
                out.TensorShape, a.TensorShape, b.TensorShape, c.TensorShape,
                paddedSizeOut, paddedSizeA, paddedSizeB, paddedSizeC,
                out.DenseMatHost + batchIdx * paddedSizeOut,
                a.DenseMatHost + (batchIdx % a.BatchSize) * paddedSizeA,
                b.DenseMatHost + (batchIdx % b.BatchSize) * paddedSizeB,
                c.DenseMatHost + (batchIdx % c.BatchSize) * paddedSizeC, 0, 2,
                Naive::Dense::NaiveGemm, M, N, paddedN, K, paddedK);
        }
    }
}

void Scale(TensorData& output, const TensorData& input, const float factor)
{
    const auto device = output.GetDevice();
    const auto M = output.Rows();
    const auto N = output.Cols();
    const auto paddedN = output.PaddedHostColSize;
    const auto stride = M * N;
    const auto strideWithPadding = M * paddedN;
    const auto totalSize = output.TensorShape.Size() * output.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;
    const auto broadcast = input.BatchSize == 1;

    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Scale(output.DenseMatCuda, input.DenseMatCuda, factor,
                           totalSize, stride, broadcast);
    }
    else
    {
        Naive::Dense::Scale(output.DenseMatHost, input.DenseMatHost, factor,
                            totalSizeWithPadding, strideWithPadding, broadcast);
    }
}

void Transpose(TensorData& output, const TensorData& input)
{
    const auto device = output.GetDevice();
    const auto M = output.Rows();
    const auto N = output.Cols();
    const auto paddedN = output.PaddedHostColSize;
    // const auto totalSize = output.TensorShape.Size() * output.BatchSize;
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