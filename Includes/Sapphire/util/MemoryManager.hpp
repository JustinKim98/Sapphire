// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_UTIL_MEMORYMANAGER_HPP
#define Sapphire_UTIL_MEMORYMANAGER_HPP

#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include <mutex>
#include <unordered_map>

namespace Sapphire::Util
{
struct pair_hash_free
{
    std::size_t operator()(const std::pair<int, size_t>& key) const
    {
        return std::hash<int>()(key.first) ^ std::hash<size_t>()(key.second);
    }
};

struct pair_hash_busy
{
    std::size_t operator()(const std::pair<int, intptr_t>& key) const
    {
        return std::hash<int>()(key.first) ^
               std::hash<uintptr_t>()(key.second);
    }
};

struct hash_convConfig
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

struct MemoryChunk
{
    MemoryChunk(size_t byteSize, void* data, int refCount)
        : ByteSize(byteSize),
          Data(data),
          RefCount(refCount)
    {
    }

    MemoryChunk(const MemoryChunk& chunk) = default;

    size_t ByteSize = 0;
    void* Data = nullptr;

    int RefCount;
};

class MemoryManager
{
public:
    //! Allocates memory on device
    //! \param byteSize : Allocation byteSize in bytes
    //! \param deviceId : device ID to allocate the memory
    static void* GetMemoryCuda(size_t byteSize, int deviceId);

    //! Allocates memory on host
    //! \param byteSize : Allocation size in bytes
    static void* GetMemoryHost(size_t byteSize);

    static Compute::Dense::Cuda::CudnnMetaData* GetCudnnMetaData(
        Compute::Dense::Cuda::ConvConfig convConfig);

    static void AddReferenceCuda(void* ptr, int deviceId);

    static void AddReferenceHost(void* ptr);

    static void AddCudnnMetaData(
        Compute::Dense::Cuda::ConvConfig convConfig,
        Compute::Dense::Cuda::CudnnMetaData* metaData);

    static void DeReferenceCuda(void* ptr, int deviceId);

    static void DeReferenceHost(void* ptr);

    static void ClearUnusedCudaMemoryPool();

    static void ClearUnusedHostMemoryPool();

    static void ClearCudaMemoryPool();

    static void ClearHostMemoryPool();

    static void ClearCudnnMetaData();

    static size_t GetTotalByteSizeCuda();

    static size_t GetTotalByteSizeHost();

    static size_t GetAllocatedByteSizeCuda();

    static size_t GetAllocatedByteSizeHost();

    static size_t GetFreeByteSizeCuda();

    static size_t GetFreeByteSizeHost();

    static bool HasConvConfig(Compute::Dense::Cuda::ConvConfig convConfig);

private:
    static std::unordered_multimap<size_t, MemoryChunk>
    m_hostFreeMemoryPool;
    static std::unordered_map<intptr_t, MemoryChunk> m_hostBusyMemoryPool;
    static std::unordered_multimap<std::pair<int, size_t>, MemoryChunk,
                                   pair_hash_free>
    m_cudaFreeMemoryPool;
    static std::unordered_map<std::pair<int, intptr_t>, MemoryChunk,
                              pair_hash_busy>
    m_cudaBusyMemoryPool;
    static std::unordered_map<Compute::Dense::Cuda::ConvConfig,
                              Compute::Dense::Cuda::CudnnMetaData*,
                              hash_convConfig>
    m_cudnnMetaData;
    static std::mutex m_hostPoolMtx;
    static std::mutex m_cudaPoolMtx;

    static unsigned int m_allocationUnitByteSize;
};
} // namespace Sapphire::Util

#endif  // Sapphire_MEMORYMANAGER_H
