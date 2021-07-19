// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/ActivationOps.hpp>
#include <Sapphire/compute/dense/cuda/Activation.cuh>
#include <Sapphire/compute/dense/naive/NaiveBasic.hpp>
#include <Sapphire/compute/dense/cuda/Basic.cuh>

namespace Sapphire::Compute
{
void SoftMax(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto unitSize = y.TensorShape.Size();
    const auto totalSize = unitSize * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::SoftMax(y.DenseMatCuda, x.DenseMatCuda, totalSize,
                             unitSize);
    }
    else
    {
        Dense::Naive::Softmax(y.DenseMatHost, x.DenseMatHost,
                              totalSizeWithPadding, unitSize, paddedN);
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
        Dense::Cuda::LeakyReLU(y.DenseMatCuda, x.DenseMatCuda, a, totalSize);
    }
    else
    {
        Dense::Naive::LeakyReLU(y.DenseMatHost, x.DenseMatHost, a,
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

void ReLUBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ReLUBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                  x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::ReLUBackward - Host not implemented");
    }
}

void LeakyReluBackward(TensorData& dx, const TensorData& dy,
                       const TensorData& x, float a)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::LeakyReLUBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                       x.DenseMatCuda,
                                       a, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::LeakyReLUBackward - Host not implemented");
    }
}
}
