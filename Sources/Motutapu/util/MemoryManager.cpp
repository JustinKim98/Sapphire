// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/Memory.cuh>
#include <Motutapu/util/MemoryManager.hpp>
#include <utility>

namespace Motutapu::Util
{
std::unordered_multimap<size_t, MemoryChunk>
    MemoryManager::m_hostFreeMemoryPool;
std::unordered_map<float*, MemoryChunk> MemoryManager::m_hostBusyMemoryPool;
std::unordered_multimap<std::pair<int, size_t>, MemoryChunk, pair_hash_free>
    MemoryManager::m_cudaFreeMemoryPool;
std::unordered_map<std::pair<int, float*>, MemoryChunk, pair_hash_busy>
    MemoryManager::m_cudaBusyMemoryPool;

std::mutex MemoryManager::m_hostPoolMtx;
std::mutex MemoryManager::m_cudaPoolMtx;

float* MemoryManager::GetMemoryCuda(size_t size, int deviceId)
{
    std::lock_guard<std::mutex> lock(m_cudaPoolMtx);
    auto success = true;
    float* cudaPtr = nullptr;

    auto key = std::make_pair(deviceId, size);
    const auto itr = m_cudaFreeMemoryPool.find(key);

    if (itr != m_cudaFreeMemoryPool.end())
    {
        auto targetChunk = itr->second;
        m_cudaFreeMemoryPool.erase(itr);
        m_cudaBusyMemoryPool.emplace(std::make_pair(deviceId, cudaPtr),
                                     targetChunk);
        return targetChunk.Data;
    }

    float* otherPtr;
    success &= Compute::Cuda::CudaSetDevice(deviceId);
    success &= Compute::Cuda::CudaMalloc(&otherPtr, size);
    MemoryChunk memoryChunk(size, cudaPtr);
    memoryChunk.deviceId = deviceId;

    m_cudaBusyMemoryPool.emplace(std::make_pair(deviceId, cudaPtr),
                                 memoryChunk);

    if (!success)
    {
        throw std::runtime_error("GetMemoryCuda - Allocation failure");
    }

    return cudaPtr;
}

float* MemoryManager::GetMemoryHost(size_t size)
{
    std::lock_guard<std::mutex> lock(m_hostPoolMtx);
    float* memory;

    const auto itr = m_hostFreeMemoryPool.find(size);
    if (itr != m_hostFreeMemoryPool.end())
    {
        auto targetChunk = itr->second;
        m_hostFreeMemoryPool.erase(itr);
        m_hostBusyMemoryPool.emplace(targetChunk.Data, targetChunk);
        return targetChunk.Data;
    }

    memory = new float[size];
    MemoryChunk memoryChunk(size, memory);
    m_hostBusyMemoryPool.emplace(memory, memoryChunk);

    return memory;
}

void MemoryManager::UnAssignMemoryCuda(float* ptr, int deviceId)
{
    std::lock_guard<std::mutex> lock(m_cudaPoolMtx);
    const auto itr = m_cudaBusyMemoryPool.find(std::make_pair(deviceId, ptr));
    if (itr == m_cudaBusyMemoryPool.end())
    {
        throw std::runtime_error(
            "UnAssignMemoryCuda - pointer has not been found in allocated "
            "pool");
    }

    auto targetChunk = itr->second;
    m_cudaBusyMemoryPool.erase(itr);
    m_cudaFreeMemoryPool.emplace(std::make_pair(deviceId, targetChunk.Size),
                                 targetChunk);
}

void MemoryManager::UnAssignMemoryHost(float* ptr)
{
    std::lock_guard<std::mutex> lock(m_hostPoolMtx);
    const auto itr = m_hostBusyMemoryPool.find(ptr);
    if (itr == m_hostBusyMemoryPool.end())
    {
        throw std::runtime_error(
            "UnAssignMemoryHost - pointer has not been found in allocated "
            "pool");
    }

    auto targetChunk = itr->second;
    m_hostBusyMemoryPool.erase(itr);
    m_hostFreeMemoryPool.emplace(targetChunk.Size, targetChunk);
}

void MemoryManager::ClearUnusedCudaMemoryPool()
{
    std::lock_guard<std::mutex> lock(m_cudaPoolMtx);

    for (auto itr = m_cudaFreeMemoryPool.begin();
         itr != m_cudaFreeMemoryPool.end(); ++itr)
    {
        if (!Compute::Cuda::CudaFree(itr->second.Data))
            throw std::runtime_error(
                "ClearUnusedCudaMemoryPool - CudaFree failed");
        m_cudaFreeMemoryPool.erase(itr);
    }
}

void MemoryManager::ClearUnusedHostMemoryPool()
{
    std::lock_guard<std::mutex> lock(m_hostPoolMtx);

    for (auto itr = m_hostFreeMemoryPool.begin();
         itr != m_hostFreeMemoryPool.end(); ++itr)
    {
        delete[] itr->second.Data;
        m_hostFreeMemoryPool.erase(itr);
    }
}

void MemoryManager::ClearCudaMemoryPool()
{
    std::lock_guard<std::mutex> lock(m_cudaPoolMtx);

    for (auto itr = m_cudaFreeMemoryPool.begin();
         itr != m_cudaFreeMemoryPool.end(); ++itr)
    {
        if (!Compute::Cuda::CudaFree(itr->second.Data))
            throw std::runtime_error(
                "ClearUnusedCudaMemoryPool - CudaFree failed");
        m_cudaFreeMemoryPool.erase(itr);
    }

    for (auto itr = m_cudaBusyMemoryPool.begin();
         itr != m_cudaBusyMemoryPool.end(); ++itr)
    {
        if (!Compute::Cuda::CudaFree(itr->second.Data))
            throw std::runtime_error(
                "ClearUnusedCudaMemoryPool - CudaFree failed");
        m_cudaBusyMemoryPool.erase(itr);
    }
}

void MemoryManager::ClearHostMemoryPool()
{
    std::lock_guard<std::mutex> lock(m_hostPoolMtx);

    for (auto itr = m_hostFreeMemoryPool.begin();
         itr != m_hostFreeMemoryPool.end(); ++itr)
    {
        delete[] itr->second.Data;
        m_hostFreeMemoryPool.erase(itr);
    }

    for (auto itr = m_hostBusyMemoryPool.begin();
         itr != m_hostBusyMemoryPool.end(); ++itr)
    {
        delete[] itr->second.Data;
        m_hostBusyMemoryPool.erase(itr);
    }
}

size_t MemoryManager::GetTotalAllocationByteSizeCuda()
{
    std::lock_guard<std::mutex> lock(m_cudaPoolMtx);

    size_t size = 0;

    for (const auto& [key, chunk] : m_cudaBusyMemoryPool)
    {
        size += chunk.Size * sizeof(float);
    }

    for (const auto& [key, chunk] : m_cudaFreeMemoryPool)
    {
        size += chunk.Size * sizeof(float);
    }

    return size;
}

size_t MemoryManager::GetTotalAllocationByteSizeHost()
{
    std::lock_guard<std::mutex> lock(m_hostPoolMtx);

    size_t size = 0;

    for (const auto& [key, chunk] : m_hostBusyMemoryPool)
    {
        size += chunk.Size * sizeof(float);
    }

    for (const auto& [key, chunk] : m_hostFreeMemoryPool)
    {
        size += chunk.Size * sizeof(float);
    }

    return size;
}

}  // namespace Motutapu::Util