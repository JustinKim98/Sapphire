// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/LossOps.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/compute/dense/naive/NaiveCrossEntropy.hpp>
#ifdef WITH_CUDA
#include <Sapphire/compute/dense/cuda/CrossEntropy.cuh>
#endif
#include <cassert>

namespace Sapphire::Compute
{
constexpr int labelIdx = 0;
constexpr int dxIdx = 0;

void CrossEntropy(TensorUtil::TensorData& y, const TensorUtil::TensorData& x,
                  const TensorUtil::TensorData& label)
{
    if (y.Mode() != x.Mode())
        throw std::runtime_error(
            "Compute::CrossEntropy - Mode of y and x was different");

    const auto batchSize = x.GetNumUnits(1);
    const auto unitSize = x.GetUnitSize(1);

    if (y.Mode() == ComputeMode::Cuda)
    {
#ifdef WITH_CUDA
        Dense::Cuda::CrossEntropy(y.CudaMutableRawPtr(), x.CudaRawPtr(),
                                  label.CudaRawPtr(), batchSize, unitSize);
#endif
    }
    else
    {
        Dense::Naive::CrossEntropy(y.HostMutableRawPtr(), x.HostRawPtr(),
                                   label.HostRawPtr(), batchSize, unitSize);
    }
}

void CrossEntropyBackward(TensorUtil::TensorData& dx,
                          const TensorUtil::TensorData& x,
                          const TensorUtil::TensorData& label)
{
    if (label.Mode() != x.Mode())
        throw std::runtime_error(
            "Compute::CrossEntropyBackward - Mode of label and x was different");

    const auto batchSize = dx.GetNumUnits(1);
    const auto unitSize = dx.GetUnitSize(1);

    if (dx.Mode() == ComputeMode::Cuda)
    {
#ifdef WITH_CUDA
        Dense::Cuda::CrossEntropyBackward(
            dx.CudaMutableRawPtr(), x.CudaMutableRawPtr(),
            label.CudaRawPtr(), batchSize,
            unitSize);
#endif
    }
    else
    {
        Dense::Naive::CrossEntropyBackward(
            dx.HostMutableRawPtr(), x.HostRawPtr(), label.HostRawPtr(),
            batchSize, unitSize);
    }
}
}
