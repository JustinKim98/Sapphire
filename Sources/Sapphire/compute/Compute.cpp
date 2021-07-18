// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/Compute.hpp>
#include <Sapphire/compute/dense/cuda/Basic.cuh>
#include <Sapphire/compute/dense/cuda/Gemm.cuh>
#include <Sapphire/compute/dense/naive/NaiveBasic.hpp>
#include <Sapphire/compute/dense/naive/NaiveGemm.hpp>
#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include <Sapphire/compute/dense/cuda/Pool.cuh>
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
            Dense::Cuda::Add(
                yElemSize * y.BatchSize,
                y.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                y.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            const auto yElemSize =
                static_cast<unsigned int>(y.GetHostElementSize());
            Dense::Naive::Add(
                yElemSize * y.BatchSize,
                y.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                yElemSize, broadcastA,
                broadcastB);
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
                             y.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             0, 1, Dense::Cuda::Add, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, y.DenseMatHost,
                             a.DenseMatHost, b.DenseMatHost, 0, 1,
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
            Dense::Cuda::Sub(y.TensorShape.Size() * y.BatchSize,
                             y.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             y.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            Dense::Naive::Sub(
                (y.TensorShape.Size() / N) * paddedN * y.BatchSize,
                y.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                (y.TensorShape.Size() / N) * paddedN, broadcastA, broadcastB);
            return;
        }
    }

    auto shapeOut = y.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    shapeOut.Expand(y.TensorShape.Dim() + 1);
    shapeA.Expand(y.TensorShape.Dim() + 1);
    shapeB.Expand(y.TensorShape.Dim() + 1);

    shapeOut.Set(0, y.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA, sizeB,
                             y.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             0, 1, Dense::Cuda::Sub, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, y.DenseMatHost,
                             a.DenseMatHost, b.DenseMatHost, 0, 1,
                             Dense::Naive::Sub, 0, false, false);
    }
}

void Gemm(TensorUtil::TensorData& y, const TensorUtil::TensorData& a,
          const TensorUtil::TensorData& b, const TensorUtil::TensorData& c)
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
        b.TensorShape.Dim() == 2 && c.TensorShape.Dim() == 2 &&
        y.BatchSize > 1)
    {
        const auto batchSize = y.BatchSize;

        if (device.Type() == DeviceType::CUDA)
        {
            Dense::Cuda::GemmMatrixWiseBroadcast(
                y.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                c.DenseMatCuda, M, N, K, batchSize, a.BatchSize == 1,
                b.BatchSize == 1, c.BatchSize == 1, 0);
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
                             sizeB, sizeC, y.DenseMatCuda, a.DenseMatCuda,
                             b.DenseMatCuda, c.DenseMatCuda, 0, 2,
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
                             y.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                             c.DenseMatHost, 0, 2, Dense::Naive::NaiveGemm, M,
                             N, paddedN, K, paddedK);
    }
}

void Conv2DForward(TensorData& y, const TensorData& x, const TensorData& filter,
                   int strideRow, int strideCol, int dilationRow,
                   int dilationCol, int rowPadding, int columnPadding)
{
    if (const auto device = y.GetDevice();
        device.Type() == DeviceType::CUDA)

    {
        const Dense::Cuda::Shape4D filterShape = {
            static_cast<int>(filter.BatchSize),
            static_cast<int>(filter.GetShape().At(2)),
            static_cast<int>(filter.GetShape().Rows()),
            static_cast<int>(filter.GetShape().Cols()) };

        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.BatchSize), static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Conv2DForward(
            y.DenseMatCuda, x.DenseMatCuda, filter.DenseMatCuda, xShape,
            filterShape, strideRow, strideCol, dilationRow, dilationCol,
            rowPadding, columnPadding, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::Conv2DForward - Host mode Not implemented");
    }
}

void MaxPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding)
{
    if (const auto device = y.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.BatchSize), static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DForward(y.DenseMatCuda, x.DenseMatCuda, xShape,
                                   windowHeight, windowWidth, strideRow,
                                   strideCol,
                                   rowPadding, columnPadding,
                                   Dense::Cuda::PoolingMode::Max,
                                   CUDNN_PROPAGATE_NAN, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::Conv2DForward - Host mode Not implemented");
    }
}


void AvgPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding)
{
    if (const auto device = y.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.BatchSize), static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DForward(
            y.DenseMatCuda, x.DenseMatCuda, xShape, windowHeight, windowWidth,
            strideRow, strideCol, rowPadding, columnPadding,
            Dense::Cuda::PoolingMode::Avg,
            CUDNN_PROPAGATE_NAN, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::Conv2DForward - Host mode Not implemented");
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
        Dense::Cuda::Scale(y.DenseMatCuda, x.DenseMatCuda, factor,
                           totalSize);
    }
    else
    {
        Dense::Naive::Scale(y.DenseMatHost, x.DenseMatHost, factor,
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
        Dense::Cuda::Transpose(y.DenseMatCuda, x.DenseMatCuda, inputM,
                               inputN, chunkSize, broadcast);
    }
    else
    {
        Dense::Naive::Transpose(y.DenseMatHost, x.DenseMatHost, inputM,
                                paddedM, inputN, paddedN, chunkSize, broadcast);
    }
}

void Dot(TensorData& y, const TensorData& a, const TensorData& b)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto broadcastA = a.BatchSize == 1;
    const auto broadcastB = b.BatchSize == 1;

    if ((y.TensorShape == a.TensorShape && y.TensorShape == b.TensorShape) &&
        y.BatchSize > 1)
    {
        if (device.Type() == DeviceType::CUDA)
        {
            Dense::Cuda::Dot(y.TensorShape.Size() * y.BatchSize, y.DenseMatCuda,
                             a.DenseMatCuda, b.DenseMatCuda,
                             y.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            Dense::Naive::Dot(
                (y.TensorShape.Size() / N) * paddedN * y.BatchSize,
                y.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                (y.TensorShape.Size() / N) * paddedN, broadcastA, broadcastB);
            return;
        }
    }

    const auto maxDim = std::max(
        { y.TensorShape.Dim(), a.TensorShape.Dim(), b.TensorShape.Dim() });

    auto shapeOut = y.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

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
                             y.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda, 0,
                             1, Dense::Cuda::Dot, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, y.DenseMatHost,
                             a.DenseMatHost, b.DenseMatHost, 0, 1,
                             Dense::Naive::Dot, 0, false, false);
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
        Dense::Cuda::Pow(y.DenseMatCuda, x.DenseMatCuda, factor,
                         totalSize);
    }
    else
    {
        Dense::Naive::Pow(y.DenseMatHost, x.DenseMatHost, factor,
                          totalSizeWithPadding);
    }
}

void cos(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::cos(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::cos(y.DenseMatHost, x.DenseMatHost,
                          totalSizeWithPadding);
    }
}

void sin(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::sin(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::sin(y.DenseMatHost, x.DenseMatHost,
                          totalSizeWithPadding);
    }
}

void tan(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::tan(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::tan(y.DenseMatHost, x.DenseMatHost,
                          totalSizeWithPadding);
    }
}

void cosh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::cosh(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::cosh(y.DenseMatHost, x.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void sinh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::sinh(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::sinh(y.DenseMatHost, x.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void tanh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::tanh(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::tanh(y.DenseMatHost, x.DenseMatHost,
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
        Dense::Cuda::log(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::log(y.DenseMatHost, x.DenseMatHost,
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
        Dense::Cuda::log10(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::log10(y.DenseMatHost, x.DenseMatHost,
                            totalSizeWithPadding);
    }
}

void ReLU(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ReLU(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::ReLU(y.DenseMatHost, x.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void ReLUBackward(TensorData& dx, const TensorData& dy)
{
    const auto device = dx.GetDevice();
    const auto N = dx.Cols();
    const auto paddedN = dx.PaddedHostColSize;
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ReLUDerivative(dx.DenseMatCuda, dy.DenseMatCuda,
                                    totalSize);
    }
    else
    {
        Dense::Naive::ReLUDerivative(dx.DenseMatHost, dy.DenseMatHost,
                                     totalSizeWithPadding);
    }
}

void LeakyReLU(TensorData& y, const TensorData& x, float a)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::LeakyReLU(y.DenseMatCuda, x.DenseMatCuda, a,
                               totalSize);
    }
    else
    {
        Dense::Naive::LeakyReLU(y.DenseMatHost, x.DenseMatHost, a,
                                totalSizeWithPadding);
    }
}

void LeakyReluBackward(TensorData& dx, const TensorData& dy, float a)
{
    const auto device = dx.GetDevice();
    const auto N = dx.Cols();
    const auto paddedN = dx.PaddedHostColSize;
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::LeakyReLUBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                       a, totalSize);
    }
    else
    {
        Dense::Naive::LeakyReLUDerivative(dx.DenseMatHost, dy.DenseMatHost,
                                          a, totalSizeWithPadding);
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
        Dense::Cuda::Inverse(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::Inverse(y.DenseMatHost, x.DenseMatHost,
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
        Dense::Cuda::Mean(y.DenseMatCuda, x.DenseMatCuda, totalSize,
                          unitSize);
    }
    else
    {
        Dense::Naive::Mean(y.DenseMatHost, x.DenseMatHost,
                           totalSizeWithPadding, unitSize);
    }
}

void Softmax(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto unitSize = y.TensorShape.Size();
    const auto totalSize = unitSize * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Softmax(y.DenseMatCuda, x.DenseMatCuda, totalSize,
                             unitSize);
    }
    else
    {
        Dense::Naive::Softmax(y.DenseMatHost, x.DenseMatHost,
                              totalSizeWithPadding, unitSize, paddedN);
    }
}

void Softmax(TensorData& dx, const TensorData& dy, const TensorData& x)
{
}
} // namespace Sapphire::Compute
