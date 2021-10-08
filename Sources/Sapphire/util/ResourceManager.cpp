// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <cassert>
#include <utility>

namespace Sapphire::Util
{
unsigned int ResourceManager::m_allocationUnitByteSize = 256;

void* ResourceManager::GetMemoryCuda(size_t byteSize, bool preserve)
{
    void* cudaPtr = nullptr;

    if (preserve)
    {
        Compute::Cuda::CudaMalloc(&cudaPtr,
                                  static_cast<unsigned int>(byteSize));
        m_cudaPreservedMemoryPool.emplace(reinterpret_cast<intptr_t>(cudaPtr),
                                          cudaPtr);
        return cudaPtr;
    }

    const auto allocationSize =
        (byteSize / m_allocationUnitByteSize) * m_allocationUnitByteSize +
        ((byteSize % m_allocationUnitByteSize) ? m_allocationUnitByteSize : 0);

    const auto itr =
        m_cudaFreeMemoryPool.find(allocationSize);

    if (itr != m_cudaFreeMemoryPool.end())
    {
        MemoryChunk targetChunk = itr->second;
        cudaPtr = targetChunk.Data;
        targetChunk.RefCount = 1;
        m_cudaFreeMemoryPool.erase(itr);
        m_cudaBusyMemoryPool.emplace(reinterpret_cast<intptr_t>(cudaPtr),
                                     targetChunk);

        return cudaPtr;
    }

    Compute::Cuda::CudaMalloc(&cudaPtr,
                              static_cast<unsigned int>(allocationSize));

    m_cudaBusyMemoryPool.emplace(reinterpret_cast<intptr_t>(cudaPtr),
                                 MemoryChunk(allocationSize, cudaPtr, 1));

    return cudaPtr;
}

void* ResourceManager::GetMemoryHost(size_t byteSize, bool preserve)
{
    void* dataPtr;

    if (preserve)
    {
#ifdef _MSC_VER
        dataPtr = _aligned_malloc(byteSize, 32);
#else
            dataPtr = aligned_alloc(32, byteSize);
#endif
        m_hostPreservedMemoryPool.emplace(
            reinterpret_cast<intptr_t>(dataPtr), dataPtr);
        return dataPtr;
    }

    const auto allocationSize =
        (byteSize / m_allocationUnitByteSize) * m_allocationUnitByteSize +
        ((byteSize % m_allocationUnitByteSize) ? m_allocationUnitByteSize : 0);

    const auto itr = m_hostFreeMemoryPool.find(allocationSize);
    if (itr != m_hostFreeMemoryPool.end())
    {
        MemoryChunk targetChunk = itr->second;
        targetChunk.RefCount = 1;
        dataPtr = targetChunk.Data;
        m_hostFreeMemoryPool.erase(itr);
        m_hostBusyMemoryPool.emplace(reinterpret_cast<intptr_t>(dataPtr),
                                     targetChunk);
        return dataPtr;
    }

#ifdef _MSC_VER
    dataPtr = _aligned_malloc(allocationSize, 32);
#else
    dataPtr = aligned_alloc(32, allocationSize);
#endif
    m_hostBusyMemoryPool.emplace(reinterpret_cast<intptr_t>(dataPtr),
                                 MemoryChunk(allocationSize, dataPtr, 1));
    return dataPtr;
}

Compute::Dense::Cuda::CudnnConv2DMetaData*
ResourceManager::GetCudnnConvMetaData(
    Compute::Dense::Cuda::ConvConfig convConfig)
{
    return m_cudnnConv2DMetaDataPool.at(convConfig);
}

Compute::Dense::Cuda::CudnnPool2DMetaData*
ResourceManager::GetCudnnPoolMetaData(
    Compute::Dense::Cuda::PoolConfig poolConfig)
{
    return m_cudnnPool2DMetaDataPool.at(poolConfig);
}

cublasHandle_t* ResourceManager::GetCublasHandle(int deviceId,
                                                 std::thread::id threadId)
{
    return m_cublasHandlePool.at(std::make_pair(deviceId, threadId));
}

cudnnHandle_t* ResourceManager::GetCudnnHandle(int deviceId,
                                               std::thread::id threadId)
{
    return m_cudnnHandlePool.at(std::make_pair(deviceId, threadId));
}

void ResourceManager::AddReferenceCuda(void* ptr)
{
    const auto itr = m_cudaBusyMemoryPool.find(reinterpret_cast<intptr_t>(ptr));
    if (itr == m_cudaBusyMemoryPool.end())
    {
        throw std::runtime_error(
            "Util::ResourceManager::AddReferenceCuda - Reference was not "
            "found");
    }

    auto& chunk = itr->second;
    chunk.RefCount += 1;
}

void ResourceManager::AddReferenceHost(void* ptr)
{
    const auto itr = m_hostBusyMemoryPool.find(reinterpret_cast<intptr_t>(ptr));
    if (itr == m_hostBusyMemoryPool.end())
    {
        throw std::runtime_error(
            "Util::ResourceManager::AddReferenceHost - Reference was not "
            "found");
    }

    auto& chunk = itr->second;
    chunk.RefCount += 1;
}

void ResourceManager::AddCublasHandle(int deviceId, std::thread::id threadId)
{
    auto* handle = new cublasHandle_t();
    cublasCreate(handle);
    m_cublasHandlePool[std::make_pair(deviceId, threadId)] = handle;
}

void ResourceManager::AddCudnnHandle(int deviceId, std::thread::id threadId)
{
    auto* handle = new cudnnHandle_t();
    cudnnCreate(handle);
    m_cudnnHandlePool[std::make_pair(deviceId, threadId)] = handle;
}

void ResourceManager::DeReferenceCuda(void* ptr, int deviceId)
{
    if (!ptr)
        throw std::runtime_error(
            "Util::ResourceManager::DereferenceCuda - Attempted to free "
            "nullptr");

    const auto itr = m_cudaBusyMemoryPool.find(reinterpret_cast<intptr_t>(ptr));

    if (itr == m_cudaBusyMemoryPool.end())
    {
        throw std::runtime_error(
            "Util::ResourceManager::DeReferenceCuda - Reference was not found");
    }

    auto& chunk = itr->second;
    chunk.RefCount -= 1;

    if (chunk.RefCount == 0)
    {
        m_cudaFreeMemoryPool.emplace(chunk.ByteSize,
                                     chunk);

        m_cudaBusyMemoryPool.erase(itr);
    }
}

void ResourceManager::DeReferenceHost(void* ptr)
{
    if (!ptr)
        throw std::runtime_error(
            "Util::ResourceManager::DeReferenceHost - Attempted to free "
            "nullptr");

    const auto itr = m_hostBusyMemoryPool.find(reinterpret_cast<intptr_t>(ptr));
    if (itr == m_hostBusyMemoryPool.end())
    {
        throw std::runtime_error(
            "Util::ResourceManager::DeReferenceHost - Reference was not found");
    }

    auto& chunk = itr->second;
    chunk.RefCount -= 1;

    if (chunk.RefCount == 0)
    {
        m_hostFreeMemoryPool.emplace(chunk.ByteSize, chunk);
        m_hostBusyMemoryPool.erase(itr);
    }
}

void ResourceManager::ClearCudnnConv2DMetaDataPool()
{
    for (auto& [key, metaData] : m_cudnnConv2DMetaDataPool)
    {
        Compute::Cuda::CudaFree(metaData->ForwardWorkSpace);
        Compute::Cuda::CudaFree(metaData->BackwardDataWorkSpace);
        Compute::Cuda::CudaFree(metaData->BackwardFilterWorkSpace);
        free(metaData);
    }

    m_cudnnConv2DMetaDataPool.clear();
}

void ResourceManager::ClearCudnnPool2DMetaDataPool()
{
    for (auto& [key, metaData] : m_cudnnPool2DMetaDataPool)
    {
        free(metaData);
    }
    m_cudnnPool2DMetaDataPool.clear();
}

void ResourceManager::ClearCublasHandlePool()
{
    for (auto& [key, handle] : m_cublasHandlePool)
    {
        cublasDestroy(*handle);
        free(handle);
    }

    m_cublasHandlePool.clear();
}

void ResourceManager::ClearCudnnHandlePool()
{
    for (auto& [key, handle] : m_cudnnHandlePool)
    {
        cudnnDestroy(*handle);
        free(handle);
    }

    m_cudnnHandlePool.clear();
}

void ResourceManager::Clean()
{
    for (auto& [key, memoryChunk] : m_hostBusyMemoryPool)
    {
        m_hostFreeMemoryPool.emplace(memoryChunk.ByteSize, memoryChunk);
    }

    for (auto& [key, memoryChunk] : m_cudaBusyMemoryPool)
    {
        m_cudaFreeMemoryPool.emplace(memoryChunk.ByteSize, memoryChunk);
    }
    m_hostBusyMemoryPool.clear();
    m_cudaBusyMemoryPool.clear();
}

void ResourceManager::ClearFreeMemoryPool()
{
    for (auto& [key, memoryChunk] : m_cudaFreeMemoryPool)
    {
        Compute::Cuda::CudaFree(memoryChunk.Data);
    }

    for (auto& [key, memoryChunk] : m_hostFreeMemoryPool)
    {
#ifdef _MSC_VER
        _aligned_free(memoryChunk.Data);
#else
        free(memoryChunk.Data);
#endif
    }

    m_hostFreeMemoryPool.clear();
    m_cudaFreeMemoryPool.clear();
}

void ResourceManager::ClearPreservedMemoryPool()
{
    for (auto& [key, ptr] : m_hostPreservedMemoryPool)
#ifdef _MSC_VER
        _aligned_free(ptr);
#else
        free(memoryChunk.Data);
#endif

    for (auto& [key, ptr] : m_cudaPreservedMemoryPool)
        Compute::Cuda::CudaFree(ptr);

    m_hostBusyMemoryPool.clear();
    m_cudaPreservedMemoryPool.clear();
}

void ResourceManager::ClearAll()
{
    ClearCudnnConv2DMetaDataPool();
    ClearCudnnPool2DMetaDataPool();
    ClearCublasHandlePool();
    ClearCudnnHandlePool();
    ClearPreservedMemoryPool();
    ClearFreeMemoryPool();
}

size_t ResourceManager::GetTotalByteSizeCuda()
{
    size_t size = 0;

    for (const auto& [key, chunk] : m_cudaBusyMemoryPool)
    {
        size += chunk.ByteSize;
    }

    for (const auto& [key, chunk] : m_cudaFreeMemoryPool)
    {
        size += chunk.ByteSize;
    }

    return size;
}

size_t ResourceManager::GetTotalByteSizeHost()
{
    size_t size = 0;

    for (const auto& [key, chunk] : m_hostBusyMemoryPool)
    {
        size += chunk.ByteSize;
    }

    for (const auto& [key, chunk] : m_hostFreeMemoryPool)
    {
        size += chunk.ByteSize;
    }

    return size;
}

bool ResourceManager::HasConvConfig(Compute::Dense::Cuda::ConvConfig convConfig)
{
    return m_cudnnConv2DMetaDataPool.find(convConfig) !=
           m_cudnnConv2DMetaDataPool.end();
}

bool ResourceManager::HasPoolConfig(Compute::Dense::Cuda::PoolConfig poolConfig)
{
    return m_cudnnPool2DMetaDataPool.find(poolConfig) !=
           m_cudnnPool2DMetaDataPool.end();
}

bool ResourceManager::HasCublasHandle(int deviceId, std::thread::id tid)
{
    return m_cublasHandlePool.find(std::make_pair(deviceId, tid)) !=
           m_cublasHandlePool.end();
}

bool ResourceManager::HasCudnnHandle(int deviceId, std::thread::id id)
{
    return m_cudnnHandlePool.find(std::make_pair(deviceId, id)) !=
           m_cudnnHandlePool.end();
}

std::unordered_multimap<size_t, MemoryChunk>
ResourceManager::m_hostFreeMemoryPool;

std::unordered_map<intptr_t, MemoryChunk> ResourceManager::m_hostBusyMemoryPool;

std::unordered_multimap<std::size_t, MemoryChunk>
ResourceManager::m_cudaFreeMemoryPool;

std::unordered_map<intptr_t, MemoryChunk, std::hash<intptr_t>>
ResourceManager::m_cudaBusyMemoryPool;

std::unordered_map<intptr_t, void*> ResourceManager::m_hostPreservedMemoryPool;
std::unordered_map<intptr_t, void*> ResourceManager::m_cudaPreservedMemoryPool;

std::unordered_map<Compute::Dense::Cuda::ConvConfig,
                   Compute::Dense::Cuda::CudnnConv2DMetaData*, ConvMetaDataHash>
ResourceManager::m_cudnnConv2DMetaDataPool;

std::unordered_map<Compute::Dense::Cuda::PoolConfig,
                   Compute::Dense::Cuda::CudnnPool2DMetaData*, PoolMetaDataHash>
ResourceManager::m_cudnnPool2DMetaDataPool;

std::unordered_map<std::pair<int, std::thread::id>, cublasHandle_t*,
                   DeviceIdTidHash>
ResourceManager::m_cublasHandlePool;

std::unordered_map<std::pair<int, std::thread::id>, cudnnHandle_t*,
                   DeviceIdTidHash>
ResourceManager::m_cudnnHandlePool;
} // namespace Sapphire::Util
