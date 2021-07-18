// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/TrigonometricOps.hpp>
#include <Sapphire/compute/dense/cuda/Trigonometric.cuh>
#include <Sapphire/compute/dense/naive/NaiveBasic.hpp>

namespace Sapphire::Compute
{
void Cos(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Cos(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::cos(y.DenseMatHost, x.DenseMatHost, totalSizeWithPadding);
    }
}

void Sin(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Sin(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::sin(y.DenseMatHost, x.DenseMatHost, totalSizeWithPadding);
    }
}

void Tan(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Tan(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::tan(y.DenseMatHost, x.DenseMatHost, totalSizeWithPadding);
    }
}

void Cosh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Cosh(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::cosh(y.DenseMatHost, x.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void Sinh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Sinh(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::sinh(y.DenseMatHost, x.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void Tanh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto paddedN = y.PaddedHostColSize;
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    const auto totalSizeWithPadding = (totalSize / N) * paddedN;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Tanh(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        Dense::Naive::tanh(y.DenseMatHost, x.DenseMatHost,
                           totalSizeWithPadding);
    }
}

void ArcCos(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcCos(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcCos - Host not implemented");
    }
}

void Arcsin(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcSin(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcSin - Host not implemented");
    }
}


void ArcTan(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcTan(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcTan - Host not implemented");
    }
}

void ArcCosh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;
    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcCosh(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcCosh - Host not implemented");
    }
}

void ArcSinh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcSinh(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcSinh - Host not implemented");
    }
}

void ArcTanh(TensorData& y, const TensorUtil::TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.TensorShape.Size() * y.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcTanh(y.DenseMatCuda, x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcTanh - Host not implemented");
    }
}
}
