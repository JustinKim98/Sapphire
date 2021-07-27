// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/Broadcast.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/dense/cuda/Basic.cuh>
#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include <Sapphire/compute/dense/cuda/Gemm.cuh>
#include <Sapphire/compute/dense/cuda/Pool.cuh>
#include <Sapphire/compute/dense/naive/NaiveBasic.hpp>
#include <Sapphire/compute/dense/naive/NaiveGemm.hpp>
#include <Sapphire/compute/dense/cuda/BasicBackward.cuh>
#include <algorithm>

namespace Sapphire::Compute
{
void Add(TensorData& y, const TensorData& a, const TensorData& b)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto broadcastA = a.BatchSize == 1;
    const auto broadcastB = b.BatchSize == 1;

    if ((y.TensorShape == a.TensorShape && y.TensorShape == b.TensorShape))
    {
        if (device.Type() == DeviceType::CUDA)
        {
            const auto yElemSize =
                static_cast<unsigned int>(y.GetCudaElementSize());
            Dense::Cuda::Add(yElemSize * y.BatchSize, y.GetMutableDenseCuda(),
                             a.GetDenseCuda(), b.GetDenseCuda(),
                             y.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            const auto yElemSize =
                static_cast<unsigned int>(y.GetHostElementSize());
            Dense::Naive::Add(yElemSize * y.BatchSize, y.GetMutableDenseHost(),
                              a.GetDenseHost(), b.GetDenseHost(), yElemSize,
                              broadcastA, broadcastB);
            return;
        }
    }

    auto shapeOut = y.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    const auto maxDim = std::max(
        { y.TensorShape.Dim(), a.TensorShape.Dim(), b.TensorShape.Dim() });

    shapeOut.Expand(maxDim + 1);
    shapeA.Expand(maxDim + 1);
    shapeB.Expand(maxDim + 1);

    shapeOut.Set(0, y.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.GetMutableDenseCuda(), a.GetDenseCuda(),
                             b.GetDenseCuda(), 0,
                             1, Dense::Cuda::Add, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, y.GetMutableDenseHost(),
                             a.GetDenseHost(), b.GetDenseHost(), 0, 1,
                             Dense::Naive::Add, 0, false, false);
    }
}

void Sub(TensorData& y, const TensorData& a, const TensorData& b)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto broadcastA = a.BatchSize == 1;
    const auto broadcastB = b.BatchSize == 1;

    if ((y.TensorShape == a.TensorShape && y.TensorShape == b.TensorShape))
    {
        if (device.Type() == DeviceType::CUDA)
        {
            const auto yElemSize =
                static_cast<unsigned int>(y.GetCudaElementSize());
            Dense::Cuda::Sub(yElemSize * y.BatchSize, y.GetMutableDenseCuda(),
                             a.GetDenseCuda(), b.GetDenseCuda(),
                             y.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            const auto yElemSize =
                static_cast<unsigned int>(y.GetHostElementSize());
            Dense::Naive::Sub(yElemSize * y.BatchSize, y.GetMutableDenseHost(),
                              a.GetDenseHost(), b.GetDenseHost(), yElemSize,
                              broadcastA, broadcastB);
            return;
        }
    }

    auto shapeOut = y.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    const auto maxDim = std::max(
        { y.TensorShape.Dim(), a.TensorShape.Dim(), b.TensorShape.Dim() });

    shapeOut.Expand(maxDim + 1);
    shapeA.Expand(maxDim + 1);
    shapeB.Expand(maxDim + 1);

    shapeOut.Set(0, y.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.GetMutableDenseCuda(), a.GetDenseCuda(),
                             b.GetDenseCuda(), 0, 1, Dense::Cuda::Sub, 0, false,
                             false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, y.GetMutableDenseHost(),
                             a.GetDenseHost(), b.GetDenseHost(), 0, 1,
                             Dense::Naive::Sub, 0, false, false);
    }
}

void Dot(TensorData& y, const TensorData& a, const TensorData& b)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto broadcastA = a.BatchSize == 1;
    const auto broadcastB = b.BatchSize == 1;

    if ((y.TensorShape == a.TensorShape && y.TensorShape == b.TensorShape))
    {
        if (device.Type() == DeviceType::CUDA)
        {
            const auto yElemSize =
                static_cast<unsigned int>(y.GetCudaElementSize());
            Dense::Cuda::Dot(yElemSize * y.BatchSize, y.GetMutableDenseCuda(),
                             a.GetDenseCuda(), b.GetDenseCuda(),
                             y.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            const auto yElemSize =
                static_cast<unsigned int>(y.GetHostElementSize());
            Dense::Naive::Dot(yElemSize * y.BatchSize, y.GetMutableDenseHost(),
                              a.GetDenseHost(), b.GetDenseHost(), yElemSize,
                              broadcastA, broadcastB);
            return;
        }
    }

    auto shapeOut = y.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    const auto maxDim = std::max(
        { y.TensorShape.Dim(), a.TensorShape.Dim(), b.TensorShape.Dim() });

    shapeOut.Expand(maxDim + 1);
    shapeA.Expand(maxDim + 1);
    shapeB.Expand(maxDim + 1);

    shapeOut.Set(0, y.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.GetMutableDenseCuda(), a.GetDenseCuda(),
                             b.GetDenseCuda(), 0, 1, Dense::Cuda::Dot, 0, false,
                             false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, y.GetMutableDenseHost(),
                             a.GetDenseHost(), b.GetDenseHost(), 0, 1,
                             Dense::Naive::Dot, 0, false, false);
    }
}

void Gemm(TensorData& y, const TensorData& a, const TensorData& b,
          const TensorData& c)
{
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
        b.TensorShape.Dim() == 2 && c.TensorShape.Dim() == 2 && y.BatchSize > 1)
    {
        const auto batchSize = y.BatchSize;

        if (device.Type() == DeviceType::CUDA)
        {
            Dense::Cuda::GemmMatrixWiseBroadcast(
                y.GetMutableDenseCuda(), a.GetDenseCuda(), b.GetDenseCuda(),
                c.GetDenseCuda(),
                M, N, K, batchSize, a.BatchSize == 1, b.BatchSize == 1,
                c.BatchSize == 1, 0);
            return;
        }
    }

    const auto maxDim = std::max({ y.TensorShape.Dim(), a.TensorShape.Dim(),
                                   b.TensorShape.Dim(), c.TensorShape.Dim() });

    //! Treat batch size as part of tensor shape
    shapeOut.Expand(maxDim + 1);
    shapeA.Expand(maxDim + 1);
    shapeB.Expand(maxDim + 1);
    shapeC.Expand(maxDim + 1);

    shapeOut.Set(0, y.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);
    shapeC.Set(0, c.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();
    const auto sizeC = shapeC.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        BroadcastWith3Inputs(shapeOut, shapeA, shapeB, shapeC, sizeOut, sizeA,
                             sizeB, sizeC, y.GetMutableDenseCuda(),
                             a.GetDenseCuda(),
                             b.GetDenseCuda(), c.GetDenseCuda(), 0, 2,
                             Dense::Cuda::Gemm, M, N, K, 0);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / K) * paddedK;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        const auto paddedSizeC = (sizeC / N) * paddedN;

        BroadcastWith3Inputs(shapeOut, shapeA, shapeB, shapeC, paddedSizeOut,
                             paddedSizeA, paddedSizeB, paddedSizeC,
                             y.GetMutableDenseHost(), a.GetDenseHost(),
                             b.GetDenseHost(),
                             c.GetDenseHost(), 0, 2, Dense::Naive::NaiveGemm, M,
                             N, paddedN, K, paddedK);
    }
}

void Scale(TensorData& y, const TensorData& x, const float factor)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Scale(y.GetMutableDenseCuda(), x.GetDenseCuda(), factor,
                           totalSize);
    }
    else
    {
        Dense::Naive::Scale(y.GetMutableDenseHost(), x.GetDenseHost(), factor,
                            totalSizeWithPadding);
    }
}

void Transpose(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto inputM = x.Rows();
    const auto inputN = x.Cols();
    const auto paddedM = y.PaddedHostColSize;
    const auto paddedN = x.PaddedHostColSize;
    const auto broadcast = x.BatchSize == 1;
    const auto chunkSize =
        y.BatchSize * y.TensorShape.Size() / (inputM * inputN);

    if (device.Type() == DeviceType::CUDA)
    {
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
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Pow(y.GetMutableDenseCuda(), x.GetDenseCuda(), factor,
                         totalSize);
    }
    else
    {
        Dense::Naive::Pow(y.GetMutableDenseHost(), x.GetDenseHost(), factor,
                          totalSizeWithPadding);
    }
}

void log(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::log(y.GetMutableDenseCuda(), x.GetDenseCuda(), totalSize);
    }
    else
    {
        Dense::Naive::log(y.GetMutableDenseHost(), x.GetDenseHost(),
                          totalSizeWithPadding);
    }
}

void log10(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::log10(y.GetMutableDenseCuda(), x.GetDenseCuda(),
                           totalSize);
    }
    else
    {
        Dense::Naive::log10(y.GetMutableDenseHost(), x.GetDenseHost(),
                            totalSizeWithPadding);
    }
}

void Inverse(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Inverse(y.GetMutableDenseCuda(), x.GetDenseCuda(),
                             totalSize);
    }
    else
    {
        Dense::Naive::Inverse(y.GetMutableDenseHost(), x.GetDenseHost(),
                              totalSizeWithPadding);
    }
}

void Mean(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto unitSize = y.TensorShape.Size();
    const auto totalSize = unitSize * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Mean(y.GetMutableDenseCuda(), x.GetDenseCuda(), totalSize,
                          unitSize);
    }
    else
    {
        Dense::Naive::Mean(y.GetMutableDenseHost(), x.GetDenseHost(),
                           totalSizeWithPadding,
                           unitSize);
    }
}

void DotBackward(TensorData& da, TensorData& db, const TensorData& dy,
                 const TensorData& a, const TensorData& b)
{
    const auto device = dy.GetDevice();

    if (dy.TensorShape == a.TensorShape && dy.TensorShape == b.TensorShape &&
        dy.BatchSize > 1)
    {
        if (device.Type() == DeviceType::CUDA)
        {
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            return;
        }
    }

    const auto maxDim = std::max(
        { dy.TensorShape.Dim(), da.TensorShape.Dim(), db.TensorShape.Dim() });

    auto shapeOut = dy.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    shapeOut.Expand(maxDim + 1);
    shapeA.Expand(maxDim + 1);
    shapeB.Expand(maxDim + 1);

    shapeOut.Set(0, dy.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (device.Type() == DeviceType::CUDA)
    {
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
} // namespace Sapphire::Compute
