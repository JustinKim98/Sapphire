// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/Initialize.hpp>
#include <Motutapu/compute/cuda/dense/Initialize.cuh>
#include <Motutapu/compute/naive/NaiveInitialize.hpp>
#include <chrono>

namespace Motutapu::Compute::Initialize
{
void Normal(const TensorUtil::TensorData& data, float mean, float sd)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Normal(data.DenseMatCuda, mean, sd,
                            data.DenseTotalLengthCuda,
                            static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Normal(data.DenseMatHost, mean, sd, data.TensorShape,
                      data.PaddedHostColSize, data.BatchSize);
    }
}

void Uniform(const TensorUtil::TensorData& data, float min, float max)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Uniform(data.DenseMatCuda, min, max,
                             data.DenseTotalLengthCuda,
                             static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Uniform(data.DenseMatHost, min, max, data.TensorShape,
                       data.PaddedHostColSize, data.BatchSize);
    }
}

void Ones(const TensorUtil::TensorData& data)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Scalar(data.DenseMatCuda, 1.0f, data.DenseTotalLengthCuda);
    }
    else
    {
        Naive::Scalar(data.DenseMatHost, 1.0f, data.TensorShape,
                      data.PaddedHostColSize, data.BatchSize);
    }
}

void Zeros(const TensorUtil::TensorData& data)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Scalar(data.DenseMatCuda, 0.0f, data.DenseTotalLengthCuda);
    }
    else
    {
        Naive::Scalar(data.DenseMatHost, 0.0f, data.TensorShape,
                      data.PaddedHostColSize, data.BatchSize);
    }
}

void HeNormal(const TensorUtil::TensorData& data, int fanIn)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Normal(
            data.DenseMatCuda, 0.0, 2.0f / std::sqrt(static_cast<float>(fanIn)),
            data.DenseTotalLengthCuda, static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Normal(data.DenseMatHost, 0.0,
                      2.0f / std::sqrt(static_cast<float>(fanIn)),
                      data.TensorShape, data.PaddedHostColSize, data.BatchSize);
    }
}

void Xavier(const TensorUtil::TensorData& data, int fanIn, int fanOut)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Normal(
            data.DenseMatCuda, 0.0,
            1.0f / std::sqrt(static_cast<float>(fanIn + fanOut)),
            data.DenseTotalLengthCuda, static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Normal(data.DenseMatHost, 0.0,
                      1.0f / std::sqrt(static_cast<float>(fanIn + fanOut)),
                      data.TensorShape, data.PaddedHostColSize, data.BatchSize);
    }
}
}  // namespace Motutapu::Compute::Initialize