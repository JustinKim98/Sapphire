// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_MEMORYMANAGER_HPP
#define MOTUTAPU_UTIL_MEMORYMANAGER_HPP

#include <atomic>
#include <cstdlib>
#include <list>
#include <mutex>
#include <unordered_map>

namespace Motutapu::Util
{
struct pair_hash_free
{
    std::size_t operator()(const std::pair<int, size_t>& pair) const
    {
        return std::hash<int>()(pair.first) ^ std::hash<size_t>()(pair.second);
    }
};

struct pair_hash_busy
{
    std::size_t operator()(const std::pair<int, float*>& pair) const
    {
        return std::hash<int>()(pair.first) ^
               std::hash<size_t>()(reinterpret_cast<size_t>(pair.second));
    }
};

struct MemoryChunk
{
    MemoryChunk(size_t size, float* data, int refCount)
        : Size(size), Data(data), RefCount(refCount)
    {
    }

    MemoryChunk(const MemoryChunk& chunk) = default;

    size_t Size = 0;
    float* Data = nullptr;

    int RefCount;
};

class MemoryManager
{
 public:
    static float* GetMemoryCuda(size_t size, int deviceId);

    static float* GetMemoryHost(size_t size);

    static void AddReferenceCuda(float* ptr, int deviceId);

    static void AddReferenceHost(float* ptr);

    static void DeReferenceCuda(float* ptr, int deviceId);

    static void DeReferenceHost(float* ptr);

    static void ClearUnusedCudaMemoryPool();

    static void ClearUnusedHostMemoryPool();

    static void ClearCudaMemoryPool();

    static void ClearHostMemoryPool();

    static size_t GetTotalAllocationByteSizeCuda();

    static size_t GetTotalAllocationByteSizeHost();

 private:
    static std::unordered_multimap<size_t, MemoryChunk> m_hostFreeMemoryPool;
    static std::unordered_map<float*, MemoryChunk> m_hostBusyMemoryPool;
    static std::unordered_multimap<std::pair<int, size_t>, MemoryChunk,
                                   pair_hash_free>
        m_cudaFreeMemoryPool;
    static std::unordered_map<std::pair<int, float*>, MemoryChunk,
                              pair_hash_busy>
        m_cudaBusyMemoryPool;

    static std::mutex m_hostPoolMtx;
    static std::mutex m_cudaPoolMtx;
};
}  // namespace Motutapu::Util

#endif  // MOTUTAPU_MEMORYMANAGER_H
