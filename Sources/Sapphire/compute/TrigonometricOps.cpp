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
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Cos(y.CudaMutableRawPtr(), x.CudaRawPtr(), totalSize);
    }
    else
    {
        Dense::Naive::Cos(y.HostMutableRawPtr(), x.HostRawPtr(),
                          totalSize);
    }
}

void Sin(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Sin(y.CudaMutableRawPtr(), x.CudaRawPtr(), totalSize);
    }
    else
    {
        Dense::Naive::Sin(y.HostMutableRawPtr(), x.HostRawPtr(),
                          totalSize);
    }
}

void Tan(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Tan(y.CudaMutableRawPtr(), x.CudaRawPtr(), totalSize);
    }
    else
    {
        Dense::Naive::Tan(y.HostMutableRawPtr(), x.HostRawPtr(),
                          totalSize);
    }
}

void Cosh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto N = y.Cols();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Cosh(y.CudaMutableRawPtr(), x.CudaRawPtr(), totalSize);
    }
    else
    {
        Dense::Naive::Cosh(y.HostMutableRawPtr(), x.HostRawPtr(),
                           totalSize);
    }
}

void Sinh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Sinh(y.CudaMutableRawPtr(), x.CudaRawPtr(), totalSize);
    }
    else
    {
        Dense::Naive::Sinh(y.HostMutableRawPtr(), x.HostRawPtr(),
                           totalSize);
    }
}

void Tanh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Tanh(y.CudaMutableRawPtr(), x.CudaRawPtr(), totalSize);
    }
    else
    {
        Dense::Naive::Tanh(y.HostMutableRawPtr(), x.HostRawPtr(),
                           totalSize);
    }
}

void ArcCos(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcCos(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                            totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcCos - Host not implemented");
    }
}

void Arcsin(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcSin(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                            totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcSin - Host not implemented");
    }
}


void ArcTan(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcTan(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                            totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcTan - Host not implemented");
    }
}

void ArcCosh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.Size();
    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcCosh(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                             totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcCosh - Host not implemented");
    }
}

void ArcSinh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcSinh(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                             totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcSinh - Host not implemented");
    }
}

void ArcTanh(TensorData& y, const TensorData& x)
{
    const auto device = y.GetDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcTanh(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                             totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::ArcTanh - Host not implemented");
    }
}

void CosBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::CosBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                 x.CudaRawPtr(), totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::CosBackward - Host not implemented");
    }
}

void SinBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::SinBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                 x.CudaRawPtr(), totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::SinBackward - Host not implemented");
    }
}

void TanBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::TanBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                 x.CudaRawPtr(), totalSize);
    }
    else
    {
        throw std::runtime_error("Compute::TanBackward - Host not implemented");
    }
}

void CoshBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    const auto device = dx.GetDevice();
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::CoshBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                  x.CudaRawPtr(), totalSize);
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
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::SinhBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                  x.CudaRawPtr(), totalSize);
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
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::TanhBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                  x.CudaRawPtr(), totalSize);
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
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcCosBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                    x.CudaRawPtr(), totalSize);
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
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcSinBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                    x.CudaRawPtr(), totalSize);
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
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcTanBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                    x.CudaRawPtr(), totalSize);
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
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcCoshBackward(dx.CudaMutableRawPtr(),
                                     dy.CudaRawPtr(),
                                     x.CudaRawPtr(), totalSize);
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
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcSinhBackward(dx.CudaMutableRawPtr(),
                                     dy.CudaRawPtr(),
                                     x.CudaRawPtr(), totalSize);
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
    const auto totalSize = dx.Size();

    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::ArcTanhBackward(dx.CudaMutableRawPtr(),
                                     dy.CudaRawPtr(),
                                     x.CudaRawPtr(), totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::ArcTanhBackward - Host not implemented");
    }
}
}
