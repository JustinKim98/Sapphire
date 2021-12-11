// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <cassert>
#include <thread>
#include <utility>

namespace Sapphire::Util
{
unsigned int ResourceManager::m_allocationUnitByteSize = 256;

void* AllocHost(std::size_t size)
{
    void* ptr = nullptr;
#ifdef _MSC_VER
    ptr = malloc(size);
    // ptr = _aligned_malloc(size, 32);
#else
    ptr = aligned_alloc(32, size);
#endif
    return ptr;
}

void FreeHost(void* ptr)
{
#ifdef _MSC_VER
    free(ptr);
    //_aligned_free(ptr);
#else
    free(ptr);
#endif
}

void* ResourceManager::GetMemoryCuda(size_t byteSize, bool preserve)
{
    void* cudaPtr = nullptr;
    const auto allocationSize =
        byteSize / m_allocationUnitByteSize * m_allocationUnitByteSize +
        (byteSize % m_allocationUnitByteSize ? m_allocationUnitByteSize : 0);
    if (preserve)
    {
        Compute::Cuda::CudaMalloc(&cudaPtr,
                                  static_cast<unsigned int>(allocationSize));
        m_cudaPreservedPool.emplace(reinterpret_cast<std::intptr_t>(cudaPtr),
                                    MemoryChunk(
                                        allocationSize, cudaPtr, 1));
        return cudaPtr;
    }

    const auto itr =
        m_cudaFreePool.find(allocationSize);

    if (itr != m_cudaFreePool.end())
    {
        cudaPtr = itr->second.Data;
        m_cudaVolatilePool.emplace(
            reinterpret_cast<std::intptr_t>(cudaPtr), itr->second);
        m_cudaFreePool.erase(itr);
    }
    else
    {
        Compute::Cuda::CudaMalloc(&cudaPtr,
                                  static_cast<unsigned int>(allocationSize));
        m_cudaVolatilePool.emplace(reinterpret_cast<std::intptr_t>(cudaPtr),
                                   MemoryChunk(allocationSize, cudaPtr, 1));
    }
    return cudaPtr;
}

void* ResourceManager::GetMemoryHost(size_t byteSize, bool preserve)
{
    void* dataPtr = nullptr;
    const auto allocationSize =
        byteSize / m_allocationUnitByteSize * m_allocationUnitByteSize +
        (byteSize % m_allocationUnitByteSize ? m_allocationUnitByteSize : 0);
    if (preserve)
    {
        dataPtr = AllocHost(allocationSize);
        m_hostPreservedPool.emplace(
            reinterpret_cast<std::intptr_t>(dataPtr),
            MemoryChunk(allocationSize, dataPtr, 1));
        return dataPtr;
    }

    const auto itr = m_hostFreePool.find(allocationSize);
    if (itr != m_hostFreePool.end())
    {
        dataPtr = itr->second.Data;
        m_hostVolatilePool.emplace(
            reinterpret_cast<std::intptr_t>(dataPtr), itr->second);
        m_hostFreePool.erase(itr);
    }
    else
    {
        dataPtr = AllocHost(allocationSize);
        m_hostVolatilePool.emplace(reinterpret_cast<std::intptr_t>(dataPtr),
                                   MemoryChunk(allocationSize, dataPtr, 1));
    }
    return dataPtr;
}

void ResourceManager::FreePreservedHost(void* ptr)
{
    auto itr =
        m_hostPreservedPool.find(reinterpret_cast<std::intptr_t>(ptr));

    if (itr == m_hostPreservedPool.end())
        throw std::runtime_error(
            "ResourceManager::FreePreservedHost - Given ptr to free was not "
            "found");
    FreeHost(ptr);
    m_hostPreservedPool.erase(reinterpret_cast<std::intptr_t>(ptr));
}

void ResourceManager::FreePreservedCuda(void* ptr)
{
    auto itr =
        m_hostPreservedPool.find(reinterpret_cast<std::intptr_t>(ptr));

    if (itr == m_hostPreservedPool.end())
        throw std::runtime_error(
            "ResourceManager::FreePreservedHost - Given ptr to free was not "
            "found");

    Compute::Cuda::CudaFree(ptr);
    m_hostPreservedPool.erase(reinterpret_cast<std::intptr_t>(ptr));
}

void ResourceManager::MoveToPreservedHost(void* ptr)
{
    auto itr = m_hostVolatilePool.find(reinterpret_cast<std::intptr_t>(ptr));
    if (itr == m_hostVolatilePool.end())
        throw std::runtime_error(
            "ResourceManager::MoveToPreservedHost - Cannot find given ptr");

    auto temp = *itr;
    m_hostVolatilePool.erase(itr);
    m_hostPreservedPool.emplace(temp);
}

void ResourceManager::MoveToPreservedCuda(void* ptr)
{
    auto itr = m_cudaVolatilePool.find(reinterpret_cast<std::intptr_t>(ptr));
    if (itr == m_cudaVolatilePool.end())
        throw std::runtime_error(
            "ResourceManager::MoveToPreservedCuda - Cannot find given ptr");

    auto temp = *itr;
    m_cudaVolatilePool.erase(itr);
    m_cudaPreservedPool.emplace(temp);
}

void ResourceManager::MoveToVolatileHost(void* ptr)
{
    auto itr = m_hostPreservedPool.find(
        reinterpret_cast<std::intptr_t>(ptr));
    if (itr == m_hostPreservedPool.end())
        throw std::runtime_error(
            "ResourceManager::MoveToPreservedHost - Cannot find given ptr");

    auto temp = *itr;
    m_hostPreservedPool.erase(itr);
    m_hostVolatilePool.emplace(temp);
}

void ResourceManager::MoveToVolatileCuda(void* ptr)
{
    auto itr =
        m_hostVolatilePool.find(reinterpret_cast<std::intptr_t>(ptr));
    if (itr == m_hostVolatilePool.end())
        throw std::runtime_error(
            "ResourceManager::MoveToPreservedHost - Cannot find given ptr");

    auto temp = *itr;
    m_hostVolatilePool.erase(itr);
    m_hostPreservedPool.emplace(temp);
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

void ResourceManager::AddCublasHandle(int deviceId, std::thread::id threadId)
{
    auto* handle = new cublasHandle_t();
    cublasCreate(handle);
    m_cublasHandlePool[std::make_pair(deviceId, threadId)] = handle;
}

void ResourceManager::AddCudnnHandle(int deviceId, std::thread::id threadId)
{
    auto* handle = new cudnnHandle_t();
    auto error = cudnnCreate(handle);
    if (error != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(
            "ResourceManager::AddCudnnHandle - Cudnn Create failed");
    m_cudnnHandlePool[std::make_pair(deviceId, threadId)] = handle;
}

void ResourceManager::ClearCudnnConv2DMetaDataPool()
{
    for (auto& [key, metaData] : m_cudnnConv2DMetaDataPool)
    {
        Compute::Cuda::CudaFree(metaData->ForwardWorkSpace);
        Compute::Cuda::CudaFree(metaData->BackwardDataWorkSpace);
        Compute::Cuda::CudaFree(metaData->BackwardFilterWorkSpace);
        delete metaData;
    }

    m_cudnnConv2DMetaDataPool.clear();
}

void ResourceManager::ClearCudnnPool2DMetaDataPool()
{
    for (auto& [key, metaData] : m_cudnnPool2DMetaDataPool)
    {
        delete metaData;
    }
    m_cudnnPool2DMetaDataPool.clear();
}

void ResourceManager::ClearCublasHandlePool()
{
    for (auto& [key, handle] : m_cublasHandlePool)
    {
        cublasDestroy(*handle);
        delete handle;
    }

    m_cublasHandlePool.clear();
}

void ResourceManager::ClearCudnnHandlePool()
{
    for (auto& [key, handle] : m_cudnnHandlePool)
    {
        cudnnDestroy(*handle);
        delete handle;
    }

    m_cudnnHandlePool.clear();
}

void ResourceManager::Clean()
{
    for (auto& [key, memoryChunk] : m_hostVolatilePool)
        m_hostFreePool.emplace(memoryChunk.ByteSize, memoryChunk);
    for (auto& [key, memoryChunk] : m_cudaVolatilePool)
        m_cudaFreePool.emplace(memoryChunk.ByteSize, memoryChunk);

    m_hostVolatilePool.clear();
    m_cudaVolatilePool.clear();
}

void ResourceManager::ClearPreservedPool()
{
    for (auto& [key, memoryChunk] : m_hostPreservedPool)
        FreeHost(memoryChunk.Data);
    for (auto& [key, memoryChunk] : m_cudaPreservedPool)
        Compute::Cuda::CudaFree(memoryChunk.Data);

    m_hostPreservedPool.clear();
    m_cudaPreservedPool.clear();
}

void ResourceManager::ClearVolatilePool()
{
    for (auto& [key, memoryChunk] : m_hostVolatilePool)
        FreeHost(memoryChunk.Data);
    for (auto& [key, memoryChunk] : m_cudaVolatilePool)
        Compute::Cuda::CudaFree(memoryChunk.Data);

    m_hostVolatilePool.clear();
    m_cudaVolatilePool.clear();
}

void ResourceManager::ClearFreePool()
{
    for (auto& [size, memoryChunk] : m_hostFreePool)
        FreeHost(memoryChunk.Data);
    for (auto& [size, memoryChunk] : m_cudaFreePool)
        Compute::Cuda::CudaFree(memoryChunk.Data);

    m_hostFreePool.clear();
    m_cudaFreePool.clear();
}

void ResourceManager::ClearAll()
{
    ClearCudnnConv2DMetaDataPool();
    ClearCudnnPool2DMetaDataPool();
    ClearCublasHandlePool();
    ClearCudnnHandlePool();
    ClearPreservedPool();
    ClearVolatilePool();
    ClearFreePool();
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

std::unordered_map<std::intptr_t, MemoryChunk>
ResourceManager::m_hostVolatilePool;

std::unordered_map<std::intptr_t, MemoryChunk>
ResourceManager::m_cudaVolatilePool;

std::unordered_multimap<std::size_t, MemoryChunk>
ResourceManager::m_hostFreePool;
std::unordered_multimap<std::size_t, MemoryChunk>
ResourceManager::m_cudaFreePool;

std::unordered_map<std::intptr_t, MemoryChunk>
ResourceManager::m_hostPreservedPool;

std::unordered_map<std::intptr_t, MemoryChunk>
ResourceManager::m_cudaPreservedPool;

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
