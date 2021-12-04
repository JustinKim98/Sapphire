// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/Broadcast.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/dense/cuda/Basic.cuh>
#include <Sapphire/compute/dense/cuda/Gemm.cuh>
#include <Sapphire/compute/dense/naive/NaiveBasic.hpp>
#include <Sapphire/compute/dense/naive/NaiveGemm.hpp>
#include <Sapphire/compute/dense/cuda/BasicBackward.cuh>
#include <Sapphire/util/UnitUtils.hpp>
#include <algorithm>
#include <cassert>


namespace Sapphire::Compute
{
void Add(TensorData& y, const TensorData& a, const TensorData& b)
{
    assert(y.Mode() == a.Mode());
    assert(y.Mode() == b.Mode());

    const auto device = y.GetDevice();

    auto shapeOut = y.GetShape();
    auto shapeA = a.GetShape();
    auto shapeB = b.GetShape();

    const auto maxDim = std::max(
        { y.GetShape().Dim(), a.GetShape().Dim(), b.GetShape().Dim() });

    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.CudaMutableRawPtr(), a.CudaRawPtr(),
                             b.CudaRawPtr(), 0,
                             0, Dense::Cuda::Add, 0, false, false);
    }
    else
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, shapeOut.Size(),
                             shapeA.Size(), shapeB.Size(),
                             y.HostMutableRawPtr(),
                             a.HostRawPtr(), b.HostRawPtr(), 0, 0,
                             Dense::Naive::Add, 0, false, false);
    }
}

void Sub(TensorData& y, const TensorData& a, const TensorData& b)
{
    assert(y.Mode() == a.Mode());
    assert(y.Mode() == b.Mode());

    const auto device = y.GetDevice();

    auto shapeOut = y.GetShape();
    auto shapeA = a.GetShape();
    auto shapeB = b.GetShape();

    const auto maxDim = std::max(
        { y.GetShape().Dim(), a.GetShape().Dim(), b.GetShape().Dim() });

    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    const int minRequiredDim = Util::GetMatchingDim(
        { shapeOut, shapeA, shapeB });

    if (y.Mode() == ComputeMode::Cuda)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.CudaMutableRawPtr(), a.CudaRawPtr(),
                             b.CudaRawPtr(), 0, minRequiredDim,
                             Dense::Cuda::Sub, 0, false,
                             false);
    }
    else
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, shapeOut.Size(),
                             shapeA.Size(), shapeB.Size(),
                             y.HostMutableRawPtr(),
                             a.HostRawPtr(), b.HostRawPtr(), 0, minRequiredDim,
                             Dense::Naive::Sub, 0, false, false);
    }
}

void Dot(TensorData& y, const TensorData& a, const TensorData& b)
{
    assert(y.Mode() == a.Mode());
    assert(y.Mode() == b.Mode());

    const auto device = y.GetDevice();

    auto shapeOut = y.GetShape();
    auto shapeA = a.GetShape();
    auto shapeB = b.GetShape();

    const auto maxDim = std::max(
        { y.GetShape().Dim(), a.GetShape().Dim(), b.GetShape().Dim() });

    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.CudaMutableRawPtr(), a.CudaRawPtr(),
                             b.CudaRawPtr(), 0, 0, Dense::Cuda::Dot, 0, false,
                             false);
    }
    else
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, shapeOut.Size(),
                             shapeA.Size(), shapeB.Size(),
                             y.HostMutableRawPtr(),
                             a.HostRawPtr(), b.HostRawPtr(), 0, 0,
                             Dense::Naive::Dot, 0, false, false);
    }
}

void DotBackward(TensorData& da, TensorData& db, const TensorData& dy,
                 const TensorData& a, const TensorData& b)
{
    assert(dy.GetDevice() == da.GetDevice());
    assert(dy.GetDevice() == db.GetDevice());
    assert(dy.GetDevice() == a.GetDevice());
    assert(dy.GetDevice() == b.GetDevice());

    const auto device = dy.GetDevice();

    const auto maxDim = std::max(
        { dy.GetShape().Dim(), da.GetShape().Dim(), db.GetShape().Dim() });

    auto shapeOut = dy.GetShape();
    auto shapeA = a.GetShape();
    auto shapeB = b.GetShape();

    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (dy.Mode() == ComputeMode::Cuda)
    {
        BroadcastBackwardWith2Inputs(
            shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB, dy.CudaRawPtr(),
            da.CudaMutableRawPtr(), db.CudaMutableRawPtr(), a.CudaRawPtr(),
            b.CudaRawPtr(), 0, 0, Dense::Cuda::DotBackward, 0, false, false);
    }
    else
    {
        throw std::runtime_error("Compute::DotBackward - Host not implemented");
    }
}

void Gemm(TensorData& y, const TensorData& a, const TensorData& b)
{
    assert(y.Mode() == a.Mode());
    assert(y.Mode() == b.Mode());

    auto shapeOut = y.GetShape();
    auto shapeA = a.GetShape();
    auto shapeB = b.GetShape();

    //! treat Make inputs, outputs to have at least 2 dimension
    shapeOut.Expand(2);
    shapeA.Expand(2);
    shapeB.Expand(2);

    const auto device = y.GetDevice();
    const auto M = shapeOut.Rows();
    const auto N = shapeOut.Cols();
    const auto K = shapeA.Cols();

    //! Faster broadcast multiply for Cuda if all tensor dimensions are fixed to
    //! 2
    if (y.GetShape().Dim() == 2 && a.GetShape().Dim() == 2 &&
        b.GetShape().Dim() == 2 && y.
        GetNumUnits(2) > 1)
    {
        const auto batchSize = y.GetNumUnits(2);

        if (y.Mode() == ComputeMode::Cuda)
        {
            Dense::Cuda::GemmMatrixWiseBroadcast(
                y.CudaMutableRawPtr(), a.CudaRawPtr(), b.CudaRawPtr(),
                M, N, K, batchSize, a.GetNumUnits(2) == 1,
                b.GetNumUnits(2) == 1, 0);
            return;
        }
    }

    const auto maxDim = std::max({ y.GetShape().Dim(), a.GetShape().Dim(),
                                   b.GetShape().Dim() });

    //! Treat batch size as part of tensor shape
    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA,
                             sizeB, y.CudaMutableRawPtr(),
                             a.CudaRawPtr(),
                             b.CudaRawPtr(), 0, 2, Dense::Cuda::Gemm, M, N, K,
                             y.GetDevice().GetID());
    }
    else
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, shapeOut.Size(),
                             shapeA.Size(), shapeB.Size(),
                             y.HostMutableRawPtr(), a.HostRawPtr(),
                             b.HostRawPtr(), 0, 2, Dense::Naive::Gemm, M,
                             N, K);
    }
}

void Scale(TensorData& y, const TensorData& x, const float factor)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto totalSize = y.GetShape().Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::Scale(y.CudaMutableRawPtr(), x.CudaRawPtr(), factor,
                           totalSize);
    }
    else
    {
        auto shapeY = y.GetShape();
        Dense::Naive::Scale(y.HostMutableRawPtr(), x.HostRawPtr(), factor,
                            shapeY.Size());
    }
}

void Transpose(TensorData& y, const TensorData& x)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto inputM = x.Rows();
    const auto inputN = x.Cols();
    const auto broadcast = x.GetNumUnits(2) == 1;
    const auto chunkSize = y.GetShape().Size() / (inputM * inputN);

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::Transpose(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                               inputM, inputN,
                               chunkSize, broadcast);
    }
    else
    {
        Dense::Naive::Transpose(y.HostMutableRawPtr(), x.HostRawPtr(),
                                inputM, inputN,
                                chunkSize, broadcast);
    }
}

//! Performs y = x^factor for each element
void Pow(TensorData& y, const TensorData& x, const float factor)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto totalSize = y.GetShape().Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::Pow(y.CudaMutableRawPtr(), x.CudaRawPtr(), factor,
                         totalSize);
    }
    else
    {
        Dense::Naive::Pow(y.HostMutableRawPtr(), x.HostRawPtr(), factor,
                          totalSize);
    }
}

void log(TensorData& y, const TensorData& x)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto totalSize = y.GetShape().Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::log(y.CudaMutableRawPtr(), x.CudaRawPtr(), totalSize);
    }
    else
    {
        Dense::Naive::log(y.HostMutableRawPtr(), x.HostRawPtr(),
                          totalSize);
    }
}

void log10(TensorData& y, const TensorData& x)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto totalSize = y.GetShape().Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::log10(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                           totalSize);
    }
    else
    {
        Dense::Naive::log10(y.HostMutableRawPtr(), x.HostRawPtr(),
                            totalSize);
    }
}

void Inverse(TensorData& y, const TensorData& x)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto totalSize = y.GetShape().Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::Inverse(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                             totalSize);
    }
    else
    {
        Dense::Naive::Inverse(y.HostMutableRawPtr(), x.HostRawPtr(),
                              totalSize);
    }
}

void Mean(TensorData& y, const TensorData& x, int dim)
{
    assert(y.Mode() == x.Mode());
    assert(y.GetShape().At(dim) == 1);

    int stride = 1;
    for (int i = dim; i < y.GetShape().Dim(); ++i)
    {
        stride *= y.GetShape().At(i);
    }

    const auto device = y.GetDevice();
    const auto unitSize = x.GetShape().At(dim);
    const auto ySize = y.GetShape().Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::Mean(y.CudaMutableRawPtr(), x.CudaRawPtr(), ySize,
                          unitSize, stride);
    }
    else
    {
        Dense::Naive::Mean(y.HostMutableRawPtr(), x.HostRawPtr(), ySize
                           , unitSize, stride);
    }
}


void MeanBackward(TensorData& dx, const TensorData& dy,
                  int dim)
{
    assert(dy.GetDevice() == dx.GetDevice());
    assert(dx.GetShape().Dim() == dy.GetShape().Dim());

    const auto device = dy.GetDevice();
    const auto yShape = dy.GetShape();
    const auto xShape = dx.GetShape();

    int stride = 1;
    for (int i = dim; i < yShape.Dim(); ++i)
    {
        stride *= yShape.At(i);
    }

    if (dy.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::MeanBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                  yShape.Size(), xShape.At(dim), stride);
    }
    else
    {
        Dense::Naive::MeanBackward(dx.HostMutableRawPtr(), dy.HostRawPtr(),
                                   yShape.Size(), xShape.At(dim), stride);
    }
}
} // namespace Sapphire::Compute
