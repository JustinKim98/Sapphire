// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/compute/cuda/dense/Gemm.cuh>
#include <Motutapu/compute/naive/NaiveGemm.hpp>
#include <cassert>

namespace Motutapu::Compute
{
[[maybe_unused]] void Gemm(TensorUtil::TensorData& out,
                           const TensorUtil::TensorData& a,
                           const TensorUtil::TensorData& b,
                           const TensorUtil::TensorData& c)
{
    const auto device = out.GetDevice();
    const auto paddedM = out.PaddedRowSize;
    const auto paddedN = out.PaddedColumnSize;
    const auto paddedK = a.PaddedColumnSize;
    const auto batchSize = out.BatchSize;
    const auto broadCastA = a.BatchSize == 1;
    const auto broadCastB = b.BatchSize == 1;
    const auto broadCastC = c.BatchSize == 1;

    if (device.Type() == DeviceType::CUDA)
    {
        Cuda::Dense::GemmCublas(out.DenseMatCuda, a.DenseMatCuda,
                                b.DenseMatCuda, c.DenseMatCuda, paddedM,
                                paddedN, paddedK, batchSize, broadCastA,
                                broadCastB, broadCastC);
    }
    else
        Naive::Dense::NaiveGemm(out.DenseMatHost, a.DenseMatHost,
                                b.DenseMatHost, c.DenseMatHost, paddedM,
                                paddedN, paddedK, batchSize, broadCastA,
                                broadCastB, broadCastC);
}

[[maybe_unused]] void Gemm(TensorUtil::TensorData& out,
                           const TensorUtil::TensorData& a,
                           const TensorUtil::TensorData& b)
{
    const auto device = out.GetDevice();
    const auto paddedM = out.PaddedRowSize;
    const auto paddedN = out.PaddedColumnSize;
    const auto paddedK = a.PaddedColumnSize;
    const auto batchSize = out.BatchSize;
    const auto broadCastA = a.BatchSize == 1;
    const auto broadCastB = b.BatchSize == 1;

    if (device.Type() == DeviceType::CUDA)
        Cuda::Dense::GemmCublas(out.DenseMatHost, a.DenseMatHost,
                                b.DenseMatHost, paddedM, paddedN, paddedK,
                                batchSize, broadCastA, broadCastB);
    else
        Naive::Dense::NaiveGemm(out.DenseMatHost, a.DenseMatHost,
                                b.DenseMatHost, out.DenseMatHost, paddedM,
                                paddedN, paddedK, batchSize, broadCastA,
                                broadCastB, false);
}

}  // namespace Motutapu::Compute