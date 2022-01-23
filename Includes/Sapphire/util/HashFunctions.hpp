// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_HASH_FUNCTIONS_HPP
#define SAPPHIRE_UTIL_HASH_FUNCTIONS_HPP

#include <thread>
#ifdef WITH_CUDA
#include <Sapphire/compute/dense/cuda/CudnnStruct.cuh>
#endif

namespace Sapphire::Util
{
struct FreePoolHash
{
    std::size_t operator()(const std::pair<int, size_t>& key) const
    {
        return std::hash<int>()(key.first) ^ std::hash<size_t>()(key.second);
    }
};

struct DeviceIdTidHash
{
    std::size_t operator()(const std::pair<int, std::thread::id>& key) const
    {
        return std::hash<int>()(key.first) ^
               std::hash<std::thread::id>()(key.second);
    }
};

#ifdef WITH_CUDA
struct ConvMetaDataHash
{
    std::size_t operator()(const Compute::Dense::Cuda::ConvConfig& key) const
    {
        const auto inputShape = key.InputShape;
        const auto filterShape = key.FilterShape;
        return std::hash<int>()(inputShape.Channels + inputShape.Height +
                                inputShape.Width) ^
               std::hash<int>()(filterShape.Channels + filterShape.Height +
                                filterShape.Width);
    }
};

struct PoolMetaDataHash
{
    std::size_t operator()(const Compute::Dense::Cuda::PoolConfig& key) const
    {
        const auto inputShape = key.InputShape;
        return std::hash<int>()(inputShape.Channels + inputShape.Height +
                                inputShape.Width) ^
               std::hash<int>()(key.WindowHeight + key.WindowWidth +
                                key.StrideRow + key.StrideCol + key.RowPadding +
                                key.ColumnPadding);
    }
};
#endif
}

#endif
