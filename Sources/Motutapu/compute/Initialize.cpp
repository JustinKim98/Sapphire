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
void Normal(TensorUtil::TensorData data, float mean, float sd)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Normal(data.DenseMatCuda, mean, sd, data.DenseTotalLength,
                            static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Normal(data.DenseMatHost, mean, sd, data.DenseTotalLength);
    }
}

void Uniform(TensorUtil::TensorData data, float min, float max)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Uniform(data.DenseMatCuda, min, max, data.DenseTotalLength,
                             static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Uniform(data.DenseMatHost, min, max, data.DenseTotalLength);
    }
}

void Ones(TensorUtil::TensorData data)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Scalar(data.DenseMatCuda, 1.0f, data.DenseTotalLength,
                            static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Scalar(data.DenseMatHost, 1.0f, data.DenseTotalLength);
    }
}

void Zeros(TensorUtil::TensorData data)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Scalar(data.DenseMatCuda, 0.0f, data.DenseTotalLength,
                            static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Scalar(data.DenseMatHost, 0.0f, data.DenseTotalLength);
    }
}

void HeNormal(TensorUtil::TensorData data, int fanIn)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Normal(
            data.DenseMatCuda, 0.0, 2.0f / std::sqrt(static_cast<float>(fanIn)),
            data.DenseTotalLength, static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Normal(data.DenseMatHost, 0.0,
                      2.0f / std::sqrt(static_cast<float>(fanIn)),
                      data.DenseTotalLength);
    }
}

void Xavier(TensorUtil::TensorData data, int fanIn, int fanOut)
{
    const auto device = data.GetDevice();
    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::Normal(
            data.DenseMatCuda, 0.0,
            1.0f / std::sqrt(static_cast<float>(fanIn + fanOut)),
            data.DenseTotalLength, static_cast<int>(std::clock()));
    }
    else
    {
        Naive::Normal(data.DenseMatHost, 0.0,
                      1.0f / std::sqrt(static_cast<float>(fanIn + fanOut)),
                      data.DenseTotalLength);
    }
}
}  // namespace Motutapu::Compute::Initialize