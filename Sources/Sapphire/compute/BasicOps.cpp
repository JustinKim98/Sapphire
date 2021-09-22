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
#include <algorithm>

namespace Sapphire::Compute
{
void Add(TensorData& y, const TensorData& a, const TensorData& b)
{
    assert(y.Mode() == a.Mode());
    assert(y.Mode() == b.Mode());

    const auto device = y.GetDevice();
    const auto paddedN = y.PaddedHostColSize;

    auto shapeOut = y.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    const auto maxDim = std::max(
        { y.TensorShape.Dim(), a.TensorShape.Dim(), b.TensorShape.Dim() });

    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.GetMutableDenseCuda(), a.GetDenseCuda(),
                             b.GetDenseCuda(), 0,
                             1, Dense::Cuda::Add, 0, false, false);
    }
    else
    {
        shapeOut.SetCol(paddedN);
        shapeA.SetCol(paddedN);
        shapeB.SetCol(paddedN);
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, shapeOut.Size(),
                             shapeA.Size(), shapeB.Size(),
                             y.GetMutableDenseHost(),
                             a.GetDenseHost(), b.GetDenseHost(), 0, 1,
                             Dense::Naive::Add, 0, false, false);
    }
}

void Sub(TensorData& y, const TensorData& a, const TensorData& b)
{
    assert(y.Mode() == a.Mode());
    assert(y.Mode() == b.Mode());

    const auto device = y.GetDevice();
    const auto paddedN = y.PaddedHostColSize;

    auto shapeOut = y.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    const auto maxDim = std::max(
        { y.TensorShape.Dim(), a.TensorShape.Dim(), b.TensorShape.Dim() });

    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.GetMutableDenseCuda(), a.GetDenseCuda(),
                             b.GetDenseCuda(), 0, 1, Dense::Cuda::Sub, 0, false,
                             false);
    }
    else
    {
        shapeOut.SetCol(paddedN);
        shapeA.SetCol(paddedN);
        shapeB.SetCol(paddedN);
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, shapeOut.Size(),
                             shapeA.Size(), shapeB.Size(),
                             y.GetMutableDenseHost(),
                             a.GetDenseHost(), b.GetDenseHost(), 0, 1,
                             Dense::Naive::Sub, 0, false, false);
    }
}

void Dot(TensorData& y, const TensorData& a, const TensorData& b)
{
    assert(y.Mode() == a.Mode());
    assert(y.Mode() == b.Mode());

    const auto device = y.GetDevice();
    const auto paddedN = y.PaddedHostColSize;

    auto shapeOut = y.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    const auto maxDim = std::max(
        { y.TensorShape.Dim(), a.TensorShape.Dim(), b.TensorShape.Dim() });

    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.GetMutableDenseCuda(), a.GetDenseCuda(),
                             b.GetDenseCuda(), 0, 1, Dense::Cuda::Dot, 0, false,
                             false);
    }
    else
    {
        shapeOut.SetCol(paddedN);
        shapeA.SetCol(paddedN);
        shapeB.SetCol(paddedN);
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, shapeOut.Size(),
                             shapeA.Size(), shapeB.Size(),
                             y.GetMutableDenseHost(),
                             a.GetDenseHost(), b.GetDenseHost(), 0, 1,
                             Dense::Naive::Dot, 0, false, false);
    }
}

void Gemm(TensorData& y, const TensorData& a, const TensorData& b,
          const TensorData& c)
{
    assert(y.Mode() == a.Mode());
    assert(y.Mode() == b.Mode());
    assert(y.Mode() == c.Mode());

    auto shapeOut = y.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;
    auto shapeC = c.TensorShape;

    //! treat Make inputs, outputs to have at least 2 dimension
    shapeOut.Expand(2);
    shapeA.Expand(2);
    shapeB.Expand(2);
    shapeC.Expand(2);

    const auto device = y.GetDevice();
    const auto M = shapeOut.Rows();
    const auto N = shapeOut.Cols();
    const auto K = shapeA.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto paddedK = a.PaddedHostColSize;

    //! Faster broadcast multiply for Cuda if all tensor dimensions are fixed to
    //! 2
    if (y.TensorShape.Dim() == 2 && a.TensorShape.Dim() == 2 &&
        b.TensorShape.Dim() == 2 && c.TensorShape.Dim() == 2 && y.
        GetBatchSize(2) > 1)
    {
        const auto batchSize = y.GetBatchSize(2);

        if (y.Mode() == DeviceType::Cuda)
        {
            cudaSetDevice(device.GetID());
            Dense::Cuda::GemmMatrixWiseBroadcast(
                y.GetMutableDenseCuda(), a.GetDenseCuda(), b.GetDenseCuda(),
                c.GetDenseCuda(),
                M, N, K, batchSize, a.GetBatchSize(2) == 1,
                b.GetBatchSize(2) == 1,
                c.GetBatchSize(2) == 1, 0);
            return;
        }
    }

    const auto maxDim = std::max({ y.TensorShape.Dim(), a.TensorShape.Dim(),
                                   b.TensorShape.Dim(), c.TensorShape.Dim() });

    //! Treat batch size as part of tensor shape
    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);
    shapeC.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();
    const auto sizeC = shapeC.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        BroadcastWith3Inputs(shapeOut, shapeA, shapeB, shapeC, sizeOut, sizeA,
                             sizeB, sizeC, y.GetMutableDenseCuda(),
                             a.GetDenseCuda(),
                             b.GetDenseCuda(), c.GetDenseCuda(), 0, 2,
                             Dense::Cuda::Gemm, M, N, K, 0);
    }
    else
    {
        shapeOut.SetCol(paddedN);
        shapeA.SetCol(paddedK);
        shapeB.SetCol(paddedN);
        shapeC.SetCol(paddedN);

        BroadcastWith3Inputs(shapeOut, shapeA, shapeB, shapeC, shapeOut.Size(),
                             shapeA.Size(), shapeB.Size(), shapeC.Size(),
                             y.GetMutableDenseHost(), a.GetDenseHost(),
                             b.GetDenseHost(),
                             c.GetDenseHost(), 0, 2, Dense::Naive::NaiveGemm, M,
                             N, paddedN, K, paddedK);
    }
}

void Scale(TensorData& y, const TensorData& x, const float factor)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Scale(y.GetMutableDenseCuda(), x.GetDenseCuda(), factor,
                           totalSize);
    }
    else
    {
        auto shapeY = y.GetShape();
        shapeY.SetCol(paddedN);
        Dense::Naive::Scale(y.GetMutableDenseHost(), x.GetDenseHost(), factor,
                            shapeY.Size(), N, paddedN);
    }
}

void Transpose(TensorData& y, const TensorData& x)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto inputM = x.Rows();
    const auto inputN = x.Cols();
    const auto paddedM = y.PaddedHostColSize;
    const auto paddedN = x.PaddedHostColSize;
    const auto broadcast = x.GetBatchSize(2) == 1;
    const auto chunkSize = y.TensorShape.Size() / (inputM * inputN);

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Transpose(y.GetMutableDenseCuda(), x.GetDenseCuda(),
                               inputM, inputN,
                               chunkSize, broadcast);
    }
    else
    {
        Dense::Naive::Transpose(y.GetMutableDenseHost(), x.GetDenseHost(),
                                inputM, paddedM,
                                inputN, paddedN, chunkSize, broadcast);
    }
}

//! Performs y = x^factor for each element
void Pow(TensorData& y, const TensorData& x, const float factor)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size();
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Pow(y.GetMutableDenseCuda(), x.GetDenseCuda(), factor,
                         totalSize);
    }
    else
    {
        Dense::Naive::Pow(y.GetMutableDenseHost(), x.GetDenseHost(), factor,
                          totalSizeWithPadding, N, paddedN);
    }
}

void log(TensorData& y, const TensorData& x)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size();
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::log(y.GetMutableDenseCuda(), x.GetDenseCuda(), totalSize);
    }
    else
    {
        Dense::Naive::log(y.GetMutableDenseHost(), x.GetDenseHost(),
                          totalSizeWithPadding, N, paddedN);
    }
}

void log10(TensorData& y, const TensorData& x)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size();
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::log10(y.GetMutableDenseCuda(), x.GetDenseCuda(),
                           totalSize);
    }
    else
    {
        Dense::Naive::log10(y.GetMutableDenseHost(), x.GetDenseHost(),
                            totalSizeWithPadding, N, paddedN);
    }
}

void Inverse(TensorData& y, const TensorData& x)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size();
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Inverse(y.GetMutableDenseCuda(), x.GetDenseCuda(),
                             totalSize);
    }
    else
    {
        Dense::Naive::Inverse(y.GetMutableDenseHost(), x.GetDenseHost(),
                              totalSizeWithPadding, N, paddedN);
    }
}

void Mean(TensorData& y, const TensorData& x, int dim)
{
    assert(y.Mode() == x.Mode());

    int stride = 1;
    for ( int i = dim; i < y.GetShape().Dim(); ++i)
    {
        stride *= y.GetShape().At(i);
    }

    const auto device = y.GetDevice();
    const auto yPaddedCols = y.PaddedHostColSize;
    const auto unitSize = x.GetShape().At(dim);
    const auto xPaddedCols = x.PaddedHostColSize;
    const auto ySize = y.GetShape().Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Mean(y.GetMutableDenseCuda(), x.GetDenseCuda(), ySize,
                          unitSize, stride);
    }
    else
    {
        Dense::Naive::Mean(y.GetMutableDenseHost(), x.GetDenseHost(), ySize
                           , unitSize, stride, y.Cols(), yPaddedCols, x.Cols(),
                           xPaddedCols);
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
        { dy.TensorShape.Dim(), da.TensorShape.Dim(), db.TensorShape.Dim() });

    auto shapeOut = dy.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    shapeOut.Expand(maxDim);
    shapeA.Expand(maxDim);
    shapeB.Expand(maxDim);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (dy.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        BroadcastBackwardWith2Inputs(
            shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB, dy.GetDenseCuda(),
            da.GetMutableDenseCuda(), db.GetMutableDenseCuda(),
            a.GetDenseCuda(),
            b.GetDenseCuda(), 0,
            0, Dense::Cuda::DotBackward, 0, false, false);
    }
    else
    {
        throw std::runtime_error("Compute::DotBackward - Host not implemented");
    }
}

void MeanBackward(TensorData& dx, const TensorData& dy, const TensorData& x,
                  int dim)
{
    assert(dy.GetDevice() == dx.GetDevice());
    assert(dy.GetDevice() == x.GetDevice());
    assert(dx.GetShape().Dim() == dy.GetShape().Dim());

    const auto device = dy.GetDevice();

    const auto yShape = dy.TensorShape;
    const auto xShape = dx.TensorShape;

    int stride = 1;
    for (int i = dim; i < yShape.Dim(); ++i)
    {
        stride *= yShape.At(i);
    }

    if (dy.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::MeanBackward(dx.GetMutableDenseCuda(), x.GetDenseCuda(),
                                  dy.GetDenseCuda(), yShape.Size(),
                                  xShape.At(dim), stride);
    }
    else
    {
        Dense::Naive::MeanBackward(dx.GetMutableDenseHost(), x.GetDenseHost(),
                                   dy.GetDenseHost(), yShape.Size(),
                                   xShape.At(dim), stride, yShape.Cols(),
                                   dy.PaddedHostColSize, xShape.Cols(),
                                   dx.PaddedHostColSize);
    }
}
} // namespace Sapphire::Compute
