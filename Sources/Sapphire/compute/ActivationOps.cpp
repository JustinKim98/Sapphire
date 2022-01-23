// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/ActivationOps.hpp>
#include <Sapphire/compute/dense/naive/NaiveBasic.hpp>
#ifdef WITH_CUDA
#include <Sapphire/compute/dense/cuda/Activation.cuh>
#include <Sapphire/compute/dense/cuda/Basic.cuh>
#endif

namespace Sapphire::Compute
{
void SoftMax(TensorData& y, const TensorData& x)
{
    if (y.GetDeviceInfo() != x.GetDeviceInfo())
        throw std::runtime_error("Compute::SoftMax - Device mismatch");

    const auto device = y.GetDeviceInfo();
    const auto unitSize = y.GetShape().At(-1);
    const auto totalSize = y.Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
#ifdef WITH_CUDA
        Dense::Cuda::SoftMax(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                             totalSize,
                             unitSize);
#endif
    }
    else
    {
        Dense::Naive::SoftMax(y.HostMutableRawPtr(), x.HostRawPtr(),
                              totalSize, unitSize);
    }
}

void LeakyReLU(TensorData& y, const TensorData& x, float a)
{
    if (y.GetDeviceInfo() != x.GetDeviceInfo())
        throw std::runtime_error("Compute::LeakyReLU - Device mismatch");

    const auto device = y.GetDeviceInfo();
    const auto totalSize = y.Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
#ifdef WITH_CUDA
        Dense::Cuda::LeakyReLU(y.CudaMutableRawPtr(), x.CudaRawPtr(), a,
                               totalSize);
#endif
    }
    else
    {
        Dense::Naive::LeakyReLU(y.HostMutableRawPtr(), x.HostRawPtr(), a,
                                totalSize);
    }
}

void ReLU(TensorData& y, const TensorData& x)
{
    if (y.GetDeviceInfo() != x.GetDeviceInfo())
        throw std::runtime_error("Compute::ReLU - Device mismatch");

    const auto device = y.GetDeviceInfo();
    const auto totalSize = y.Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
#ifdef WITH_CUDA
        Dense::Cuda::ReLU(y.CudaMutableRawPtr(), x.CudaRawPtr(), totalSize);
#endif
    }
    else
    {
        Dense::Naive::ReLU(y.HostMutableRawPtr(), x.HostRawPtr(),
                           totalSize);
    }
}

void ReLUBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    if (dx.GetDeviceInfo() != dy.GetDeviceInfo() || dx.GetDeviceInfo() != x.
        GetDeviceInfo())
        throw std::runtime_error("Compute::ReLUBackward - Device mismatch");

    const auto device = dx.GetDeviceInfo();
    const auto totalSize = dx.Size();

    if (dx.Mode() == ComputeMode::Cuda)
    {
#ifdef WITH_CUDA
        Dense::Cuda::ReLUBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                  x.CudaRawPtr(), totalSize);
#endif
    }
    else
    {
        Dense::Naive::ReLUBackward(dx.HostMutableRawPtr(), dy.HostRawPtr(),
                                   x.HostRawPtr(), totalSize);
    }
}

void LeakyReLUBackward(TensorData& dx, const TensorData& dy,
                       const TensorData& x, float a)
{
    if (dx.GetDeviceInfo() != dy.GetDeviceInfo() || dx.GetDeviceInfo() != x.
        GetDeviceInfo())
        throw std::runtime_error(
            "Compute::LeakyReLUBackward - Device mismatch");

    const auto device = dx.GetDeviceInfo();

    if (dx.Mode() == ComputeMode::Cuda)
    {
#ifdef WITH_CUDA
        const auto totalSize = dx.Size();
        Dense::Cuda::LeakyReLUBackward(dx.CudaMutableRawPtr(),
                                       dy.CudaRawPtr(),
                                       x.CudaRawPtr(),
                                       a, totalSize);
#endif
    }
    else
    {
        throw std::runtime_error(
            "Compute::LeakyReLUBackward - Host not implemented");
    }
}

void SoftMaxBackward(TensorData& dx, const TensorData& dy, const TensorData& y)
{
    if (dx.GetDeviceInfo() != dy.GetDeviceInfo() || dx.GetDeviceInfo() != y.
        GetDeviceInfo())
        throw std::runtime_error("Compute::SoftMaxBackward - Device mismatch");

    const auto device = dx.GetDeviceInfo();
    const auto totalSize = dx.Size();
    const auto unitSize = dx.GetShape().At(-1);

    if (dx.Mode() == ComputeMode::Cuda)
    {
#ifdef WITH_CUDA
        Dense::Cuda::SoftmaxBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                     y.CudaRawPtr(), totalSize, unitSize);
#endif
    }
    else
    {
        Dense::Naive::SoftMaxBackward(dx.HostMutableRawPtr(), dy.HostRawPtr(),
                                      y.HostRawPtr(), totalSize, unitSize);
    }
}
}
