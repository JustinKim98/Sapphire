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
    assert(y.Mode() == x.Mode());
    const auto device = y.GetCudaDevice();
    const auto unitSize = y.GetShape().At(-1);
    const auto totalSize = y.Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::SoftMax(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                             totalSize,
                             unitSize);
    }
    else
    {
        Dense::Naive::SoftMax(y.HostMutableRawPtr(), x.HostRawPtr(),
                              totalSize, unitSize);
    }
}

void LeakyReLU(TensorData& y, const TensorData& x, float a)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetCudaDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::LeakyReLU(y.CudaMutableRawPtr(), x.CudaRawPtr(), a,
                               totalSize);
    }
    else
    {
        Dense::Naive::LeakyReLU(y.HostMutableRawPtr(), x.HostRawPtr(), a,
                                totalSize);
    }
}

void ReLU(TensorData& y, const TensorData& x)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetCudaDevice();
    const auto totalSize = y.Size();

    if (y.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::ReLU(y.CudaMutableRawPtr(), x.CudaRawPtr(), totalSize);
    }
    else
    {
        Dense::Naive::ReLU(y.HostMutableRawPtr(), x.HostRawPtr(),
                           totalSize);
    }
}

void ReLUBackward(TensorData& dx, const TensorData& dy, const TensorData& x)
{
    assert(dx.Mode() == dy.Mode() && dx.Mode() == x.Mode());
    const auto device = dx.GetCudaDevice();
    const auto totalSize = dx.Size();

    if (dx.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::ReLUBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                  x.CudaRawPtr(), totalSize);
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
    const auto device = dx.GetCudaDevice();
    const auto totalSize = dx.Size();

    if (dx.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::LeakyReLUBackward(dx.CudaMutableRawPtr(),
                                       dy.CudaRawPtr(),
                                       x.CudaRawPtr(),
                                       a, totalSize);
    }
    else
    {
        throw std::runtime_error(
            "Compute::LeakyReLUBackward - Host not implemented");
    }
}

void SoftMaxBackward(TensorData& dx, const TensorData& dy, const TensorData& y)
{
    assert(dx.Mode() == dy.Mode() && dx.Mode() == y.Mode());
    const auto device = dx.GetCudaDevice();
    const auto totalSize = dx.Size();
    const auto unitSize = dx.GetShape().At(-1);

    if (dx.Mode() == ComputeMode::Cuda)
    {
        Dense::Cuda::SoftmaxBackward(dx.CudaMutableRawPtr(), dy.CudaRawPtr(),
                                     y.CudaRawPtr(), totalSize, unitSize);
    }
    else
    {
        Dense::Naive::SoftMaxBackward(dx.HostMutableRawPtr(), dy.HostRawPtr(),
                                      y.HostRawPtr(), totalSize, unitSize);
    }
}
}
