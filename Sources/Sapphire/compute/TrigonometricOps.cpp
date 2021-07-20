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

void ArcTanh(TensorData& y, const TensorData& x)
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

void CosBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::CosBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                 x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::CosBackward - Host not implemented");
    }
}

void SinBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::SinBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                 x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::SinBackward - Host not implemented");
    }
}

void TanBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::TanBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                 x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::TanBackward - Host not implemented");
    }
}

void CoshBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::CoshBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                  x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::CoshBackward - Host not implemented");
    }
}

void SinhBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::SinhBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                  x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::SinhBackward - Host not implemented");
    }
}

void TanhBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::TanhBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                  x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::TanhBackward - Host not implemented");
    }
}

void ArcCosBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcCosBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                    x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::ArcCosBackward - Host not implemented");
    }
}

void ArcSinBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcSinBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                    x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::ArcSinBackward - Host not implemented");
    }
}

void ArcTanBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcTanBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                    x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::ArcTanBackward- Host not implemented");
    }
}

void ArcCoshBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcCoshBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                     x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::ArcCoshBackward - Host not implemented");
    }
}

void ArcSinhBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcSinhBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                     x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::ArcSinhBackward - Host not implemented");
    }
}

void ArcTanhBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.TensorShape.Size() * dx.BatchSize;

    if (device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::ArcTanhBackward(dx.DenseMatCuda, dy.DenseMatCuda,
                                     x.DenseMatCuda, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::ArcTanhBackward - Host not implemented");
    }
}
}
