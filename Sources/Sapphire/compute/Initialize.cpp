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
        Dense::Cuda::Normal(data.GetMutableDenseCuda(), mean, sd,
                            data.DenseTotalLengthCuda,
                            static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Normal(data.GetMutableDenseHost(), mean, sd,
                             data.TensorShape,
                             data.PaddedHostColSize);
    }
}

void Uniform(TensorUtil::TensorData& data, float min, float max)
{
    if (const auto device = data.GetDevice();
        data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Uniform(data.GetMutableDenseCuda(), min, max,
                             data.DenseTotalLengthCuda,
                             static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Uniform(data.GetMutableDenseHost(), min, max,
                              data.TensorShape,
                              data.PaddedHostColSize);
    }
}

void Ones(TensorUtil::TensorData& data)
{
    const auto device = data.GetDevice();
    if (data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Scalar(data.GetMutableDenseCuda(), 1.0f,
                            data.DenseTotalLengthCuda);
    }
    else
    {
        Dense::Naive::Scalar(data.GetMutableDenseHost(), 1.0f, data.TensorShape,
                             data.PaddedHostColSize);
    }
}

void Zeros(TensorUtil::TensorData& data)
{
    const auto device = data.GetDevice();
    if (data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Scalar(data.GetMutableDenseCuda(), 0.0f,
                            data.DenseTotalLengthCuda);
    }
    else
    {
        Dense::Naive::Scalar(data.GetMutableDenseHost(), 0.0f, data.TensorShape,
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
            data.GetMutableDenseCuda(), 0.0,
            2.0f / std::sqrt(static_cast<float>(fanIn)),
            data.DenseTotalLengthCuda, static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Normal(data.GetMutableDenseHost(), 0.0,
                             2.0f / std::sqrt(static_cast<float>(fanIn)),
                             data.TensorShape, data.PaddedHostColSize);
    }
}

void Xavier(TensorUtil::TensorData& data, int fanIn, int fanOut)
{
    const auto device = data.GetDevice();
    if (data.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        Dense::Cuda::Normal(
            data.GetMutableDenseCuda(), 0.0,
            1.0f / std::sqrt(static_cast<float>(fanIn + fanOut)),
            data.DenseTotalLengthCuda, static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Normal(data.GetMutableDenseHost(), 0.0,
                             1.0f / std::sqrt(
                                 static_cast<float>(fanIn + fanOut)),
                             data.TensorShape, data.PaddedHostColSize);
    }
}
} // namespace Sapphire::Compute::Initialize
