// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/compute/dense/cuda/Initialize.cuh>
#include <Sapphire/compute/dense/naive/NaiveInitialize.hpp>
#include <chrono>
#include <cmath>

namespace Sapphire::Compute::Initialize
{
void Normal(TensorUtil::TensorData& data, float mean, float sd)
{
    const auto device = data.GetDevice();
    if (data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Normal(data.CudaMutableRawPtr(), mean, sd,
                            data.DenseTotalLengthCuda,
                            static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Normal(data.HostMutableRawPtr(), mean, sd,
                             data.GetShape(),
                             data.PaddedHostColSize);
    }
}

void Uniform(TensorUtil::TensorData& data, float min, float max)
{
    if (const auto device = data.GetDevice();
        data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Uniform(data.CudaMutableRawPtr(), min, max,
                             data.DenseTotalLengthCuda,
                             static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Uniform(data.HostMutableRawPtr(), min, max,
                              data.GetShape(),
                              data.PaddedHostColSize);
    }
}

void Ones(TensorUtil::TensorData& data)
{
    const auto device = data.GetDevice();
    if (data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Scalar(data.CudaMutableRawPtr(), 1.0f,
                            data.DenseTotalLengthCuda);
    }
    else
    {
        Dense::Naive::Scalar(data.HostMutableRawPtr(), 1.0f, data.GetShape(),
                             data.PaddedHostColSize);
    }
}

void Zeros(TensorUtil::TensorData& data)
{
    const auto device = data.GetDevice();
    if (data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Scalar(data.CudaMutableRawPtr(), 0.0f,
                            data.DenseTotalLengthCuda);
    }
    else
    {
        Dense::Naive::Scalar(data.HostMutableRawPtr(), 0.0f, data.GetShape(),
                             data.PaddedHostColSize);
    }
}

void Scalar(TensorUtil::TensorData& data, float value)
{
    const auto device = data.GetDevice();
    if (data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Scalar(data.CudaMutableRawPtr(), value,
                            data.DenseTotalLengthCuda);
    }
    else
    {
        Dense::Naive::Scalar(data.HostMutableRawPtr(), value, data.GetShape(),
                             data.PaddedHostColSize);
    }
}

void HeNormal(TensorUtil::TensorData& data, int fanIn)
{
    const auto device = data.GetDevice();
    if (data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Normal(
            data.CudaMutableRawPtr(), 0.0,
            2.0f / std::sqrt(static_cast<float>(fanIn)),
            data.DenseTotalLengthCuda, static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Normal(data.HostMutableRawPtr(), 0.0,
                             2.0f / std::sqrt(static_cast<float>(fanIn)),
                             data.GetShape(), data.PaddedHostColSize);
    }
}

void Xavier(TensorUtil::TensorData& data, int fanIn, int fanOut)
{
    const auto device = data.GetDevice();
    if (data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Normal(
            data.CudaMutableRawPtr(), 0.0,
            1.0f / std::sqrt(static_cast<float>(fanIn + fanOut)),
            data.DenseTotalLengthCuda, static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Normal(data.HostMutableRawPtr(), 0.0,
                             1.0f / std::sqrt(
                                 static_cast<float>(fanIn + fanOut)),
                             data.GetShape(), data.PaddedHostColSize);
    }
}
} // namespace Sapphire::Compute::Initialize
