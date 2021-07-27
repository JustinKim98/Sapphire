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
    if (const auto device = data.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Normal(data.GetMutableDenseCuda(), mean, sd,
                            data.DenseTotalLengthCuda,
                            static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Normal(data.GetMutableDenseHost(), mean, sd,
                             data.TensorShape,
                             data.PaddedHostColSize, data.BatchSize);
    }
}

void Uniform(TensorUtil::TensorData& data, float min, float max)
{
    if (const auto device = data.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Uniform(data.GetMutableDenseCuda(), min, max,
                             data.DenseTotalLengthCuda,
                             static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Uniform(data.GetMutableDenseHost(), min, max,
                              data.TensorShape,
                              data.PaddedHostColSize, data.BatchSize);
    }
}

void Ones(TensorUtil::TensorData& data)
{
    if (const auto device = data.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Scalar(data.GetMutableDenseCuda(), 1.0f,
                            data.DenseTotalLengthCuda);
    }
    else
    {
        Dense::Naive::Scalar(data.GetMutableDenseHost(), 1.0f, data.TensorShape,
                             data.PaddedHostColSize, data.BatchSize);
    }
}

void Zeros(TensorUtil::TensorData& data)
{
    if (const auto device = data.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Scalar(data.GetMutableDenseCuda(), 0.0f,
                            data.DenseTotalLengthCuda);
    }
    else
    {
        Dense::Naive::Scalar(data.GetMutableDenseHost(), 0.0f, data.TensorShape,
                             data.PaddedHostColSize, data.BatchSize);
    }
}

void HeNormal(TensorUtil::TensorData& data, int fanIn)
{
    if (const auto device = data.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        Dense::Cuda::Normal(
            data.GetMutableDenseCuda(), 0.0,
            2.0f / std::sqrt(static_cast<float>(fanIn)),
            data.DenseTotalLengthCuda, static_cast<int>(std::clock()));
    }
    else
    {
        Dense::Naive::Normal(data.GetMutableDenseHost(), 0.0,
                             2.0f / std::sqrt(static_cast<float>(fanIn)),
                             data.TensorShape, data.PaddedHostColSize,
                             data.BatchSize);
    }
}

void Xavier(TensorUtil::TensorData& data, int fanIn, int fanOut)
{
    if (const auto device = data.GetDevice(); device.Type() == DeviceType::CUDA)
    {
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
                             data.TensorShape, data.PaddedHostColSize,
                             data.BatchSize);
    }
}
} // namespace Sapphire::Compute::Initialize
