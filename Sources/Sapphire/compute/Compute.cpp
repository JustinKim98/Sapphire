// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/Compute.hpp>
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <Sapphire/compute/dense/cuda/Basic.cuh>
#include <Sapphire/compute/dense/cuda/Gemm.cuh>
#include <Sapphire/compute/dense/naive/NaiveBasic.hpp>
#include <Sapphire/compute/dense/naive/NaiveGemm.hpp>
#include <algorithm>

namespace Sapphire::Compute
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
            Dense::Cuda::Add(out.TensorShape.Size() * out.BatchSize,
                             out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             out.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            Dense::Naive::Add(
                (out.TensorShape.Size() / N) * paddedN * out.BatchSize,
                out.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                (out.TensorShape.Size() / N) * paddedN, broadcastA, broadcastB);
            return;
        }
    }

    auto shapeOut = out.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    const auto maxDim = std::max(
        { out.TensorShape.Dim(), a.TensorShape.Dim(), b.TensorShape.Dim() });

    shapeOut.Expand(maxDim + 1);
    shapeA.Expand(maxDim + 1);
    shapeB.Expand(maxDim + 1);

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
                             0, 0, Dense::Cuda::Add, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, out.DenseMatHost,
                             a.DenseMatHost, b.DenseMatHost, 0, 0,
                             Dense::Naive::Add, 0, false, false);
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
            Dense::Cuda::Sub(out.TensorShape.Size() * out.BatchSize,
                             out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             out.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            Dense::Naive::Sub(
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
                             0, 0, Dense::Cuda::Sub, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, out.DenseMatHost,
                             a.DenseMatHost, b.DenseMatHost, 0, 1,
                             Dense::Naive::Sub, 0, false, false);
    }
}

void Gemm(TensorUtil::TensorData& out, const TensorUtil::TensorData& a,
          const TensorUtil::TensorData& b, const TensorUtil::TensorData& c)
{
    auto shapeOut = out.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;
    auto shapeC = c.TensorShape;

    //! treat Make inputs, outputs to have at least 2 dimension
    shapeOut.Expand(2);
    shapeA.Expand(2);
    shapeB.Expand(2);
    shapeC.Expand(2);

    const auto device = out.GetDevice();
    const auto M = shapeOut.Rows();
    const auto N = shapeOut.Cols();
    const auto K = shapeA.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto paddedK = a.PaddedHostColSize;

    //! Faster broadcast multiply for Cuda if all tensor dimensions are fixed to
    //! 2
    if (out.TensorShape.Dim() == 2 && a.TensorShape.Dim() == 2 &&
        b.TensorShape.Dim() == 2 && c.TensorShape.Dim() == 2 &&
        out.BatchSize > 1)
    {
        const auto batchSize = out.BatchSize;

        if (device.Type() == DeviceType::CUDA)
        {
            Dense::Cuda::GemmMatrixWiseBroadcast(
                out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                c.DenseMatCuda, M, N, K, batchSize, a.BatchSize == 1,
                b.BatchSize == 1, c.BatchSize == 1);
            return;
        }
    }

    const auto maxDim = std::max({ out.TensorShape.Dim(), a.TensorShape.Dim(),
                                   b.TensorShape.Dim(), c.TensorShape.Dim() });

    //! Treat batch size as part of tensor shape
    shapeOut.Expand(maxDim + 1);
    shapeA.Expand(maxDim + 1);
    shapeB.Expand(maxDim + 1);
    shapeC.Expand(maxDim + 1);

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
                             Dense::Cuda::Gemm, M, N, K, &cublasHandle);

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
                             c.DenseMatHost, 0, 2, Dense::Naive::NaiveGemm, M,
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
        Dense::Cuda::Scale(output.DenseMatCuda, input.DenseMatCuda, factor,
                           totalSize);
    }
    else
    {
        Dense::Naive::Scale(output.DenseMatHost, input.DenseMatHost, factor,
                            totalSizeWithPadding);
    }
}

void Transpose(TensorData& output, const TensorData& input)
{
    const auto device = output.GetDevice();
    const auto inputM = input.Rows();
    const auto inputN = input.Cols();
    const auto paddedM = output.PaddedHostColSize;
    const auto paddedN = input.PaddedHostColSize;
    const auto broadcast = input.BatchSize == 1;
    const auto chunkSize =
        output.BatchSize * output.TensorShape.Size() / (inputM * inputN);

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Transpose(output.DenseMatCuda, input.DenseMatCuda, inputM,
                               inputN, chunkSize, broadcast);
    }
    else
    {
        Dense::Naive::Transpose(output.DenseMatHost, input.DenseMatHost, inputM,
                                paddedM, inputN, paddedN, chunkSize, broadcast);
    }
}

void Dot(TensorData& out, const TensorData& a, const TensorData& b)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto broadcastA = a.BatchSize == 1;
    const auto broadcastB = b.BatchSize == 1;

    if ((out.TensorShape == a.TensorShape &&
         out.TensorShape == b.TensorShape) &&
        out.BatchSize > 1)
    {
        if (device.Type() == DeviceType::CUDA)
        {
            Dense::Cuda::Dot(out.TensorShape.Size() * out.BatchSize,
                             out.DenseMatCuda, a.DenseMatCuda, b.DenseMatCuda,
                             out.TensorShape.Size(), broadcastA, broadcastB);
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            Dense::Naive::Dot(
                (out.TensorShape.Size() / N) * paddedN * out.BatchSize,
                out.DenseMatHost, a.DenseMatHost, b.DenseMatHost,
                (out.TensorShape.Size() / N) * paddedN, broadcastA, broadcastB);
            return;
        }
    }

    const auto maxDim = std::max(
        { out.TensorShape.Dim(), a.TensorShape.Dim(), b.TensorShape.Dim() });

    auto shapeOut = out.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    shapeOut.Expand(maxDim + 1);
    shapeA.Expand(maxDim + 1);
    shapeB.Expand(maxDim + 1);

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
                             0, 0, Dense::Cuda::Dot, 0, false, false);
    }
    else
    {
        const auto paddedSizeOut = (sizeOut / N) * paddedN;
        const auto paddedSizeA = (sizeA / N) * paddedN;
        const auto paddedSizeB = (sizeB / N) * paddedN;
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, paddedSizeOut,
                             paddedSizeA, paddedSizeB, out.DenseMatHost,
                             a.DenseMatHost, b.DenseMatHost, 0, 1,
                             Dense::Naive::Dot, 0, false, false);
    }
}

//! Performs out = input^factor for each element
void Pow(TensorData& out, const TensorData& input, const float factor)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Pow(out.DenseMatCuda, input.DenseMatCuda, factor,
                         totalSize);
    }
    else
    {
        Dense::Naive::Pow(out.DenseMatHost, input.DenseMatHost, factor,
                          totalSizeWithPadding);
    }
}

void cos(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::cos(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::cos(out.DenseMatHost, input.DenseMatHost,
                          totalSizeWithPadding);
    }
}

void sin(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::sin(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::sin(out.DenseMatHost, input.DenseMatHost,
                          totalSizeWithPadding);
    }
}

void tan(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::tan(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::tan(out.DenseMatHost, input.DenseMatHost,
                          totalSizeWithPadding);
    }
}

void cosh(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::cosh(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::cosh(out.DenseMatHost, input.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void sinh(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::sinh(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::sinh(out.DenseMatHost, input.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void tanh(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::tanh(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::tanh(out.DenseMatHost, input.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void log(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::log(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::log(out.DenseMatHost, input.DenseMatHost,
                          totalSizeWithPadding);
    }
}

void log10(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::log10(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::log10(out.DenseMatHost, input.DenseMatHost,
                            totalSizeWithPadding);
    }
}

void ReLU(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ReLU(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::ReLU(out.DenseMatHost, input.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void ReLUDerivative(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ReLUDerivative(out.DenseMatCuda, input.DenseMatCuda,
                                    totalSize);
    }
    else
    {
        Dense::Naive::ReLUDerivative(out.DenseMatHost, input.DenseMatHost,
                                     totalSizeWithPadding);
    }
}

void LeakyReLU(TensorData& out, const TensorData& input, float a)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::LeakyReLU(out.DenseMatCuda, input.DenseMatCuda, a,
                               totalSize);
    }
    else
    {
        Dense::Naive::LeakyReLU(out.DenseMatHost, input.DenseMatHost, a,
                                totalSizeWithPadding);
    }
}

void LeakyReluDerivative(TensorData& out, const TensorData& input, float a)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::LeakyReLUDerivative(out.DenseMatCuda, input.DenseMatCuda,
                                         a, totalSize);
    }
    else
    {
        Dense::Naive::LeakyReLUDerivative(out.DenseMatHost, input.DenseMatHost,
                                          a, totalSizeWithPadding);
    }
}

void Inverse(TensorData& out, const TensorData& input)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto totalSize = out.TensorShape.Size() * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Inverse(out.DenseMatCuda, input.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::Inverse(out.DenseMatHost, input.DenseMatHost,
                              totalSizeWithPadding);
    }
}

void Mean(TensorData& out, const TensorData& x)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto unitSize = out.TensorShape.Size();
    const auto totalSize = unitSize * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Mean(out.DenseMatCuda, x.DenseMatCuda, totalSize,
                          unitSize);
    }
    else
    {
        Dense::Naive::Mean(out.DenseMatHost, x.DenseMatHost,
                           totalSizeWithPadding, unitSize);
    }
}

void Softmax(TensorData& out, const TensorData& x)
{
    const auto device = out.GetDevice();
    const auto N = out.Cols();
    const auto paddedN = out.PaddedHostColSize;
    const auto unitSize = out.TensorShape.Size();
    const auto totalSize = unitSize * out.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Softmax(out.DenseMatCuda, x.DenseMatCuda, totalSize,
                             unitSize);
    }
    else
    {
        Dense::Naive::Softmax(out.DenseMatHost, x.DenseMatHost,
                              totalSizeWithPadding, unitSize, paddedN);
    }
}

}  // namespace Sapphire::Compute