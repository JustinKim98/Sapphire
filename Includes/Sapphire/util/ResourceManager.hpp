// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_MEMORYMANAGER_HPP
#define SAPPHIRE_UTIL_MEMORYMANAGER_HPP

#include <Sapphire/compute/dense/cuda/CudnnStruct.cuh>
#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include <Sapphire/compute/dense/cuda/Pool.cuh>
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <mutex>
#include <thread>
#include <unordered_map>
//#include <Sapphire/operations/Backward/BackPropWrapperKey.hpp>

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
    std::size_t operator()(
        const std::pair<int, std::thread::id>& key) const
    {
        return std::hash<int>()(key.first) ^
               std::hash<std::thread::id>()(key.second);
    }
};

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
                                inputShape.Width) ^ std::hash<int>()(
                   key.WindowHeight + key.WindowWidth +
                   key.StrideRow + key.StrideCol + key.RowPadding + key.
                   ColumnPadding);
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

    ~MemoryChunk() = default;
    MemoryChunk(const MemoryChunk& chunk) = default;
    MemoryChunk(MemoryChunk&& chunk) = default;
    MemoryChunk& operator=(const MemoryChunk& chunk) = default;
    MemoryChunk& operator=(MemoryChunk&& chunk) noexcept = default;

    size_t ByteSize = 0;
    void* Data = nullptr;

    int RefCount;
};

class ResourceManager
{
public:
    //! Allocates memory on device
    //! \param byteSize : Allocation byteSize in bytes

    static void* GetMemoryCuda(size_t byteSize,
                               bool preserve = false);

    //! Allocates memory on host
    //! \param byteSize : Allocation size in bytes
    static void* GetMemoryHost(size_t byteSize, bool preserve = false);

    static Compute::Dense::Cuda::CudnnConv2DMetaData* GetCudnnConvMetaData(
        Compute::Dense::Cuda::ConvConfig convConfig);

    static Compute::Dense::Cuda::CudnnPool2DMetaData* GetCudnnPoolMetaData(
        Compute::Dense::Cuda::PoolConfig poolConfig);

    static cublasHandle_t* GetCublasHandle(int deviceId,
                                           std::thread::id threadId);

    static cudnnHandle_t*
    GetCudnnHandle(int deviceId, std::thread::id threadId);

    static void AddReferenceCuda(void* ptr);

    static void AddReferenceHost(void* ptr);

    template <typename ...Ts>
    static void AddCudnnConv2DMetaData(
        Compute::Dense::Cuda::ConvConfig convConfig, Ts ... args)
    {
        auto* metaData = new Compute::Dense::Cuda::CudnnConv2DMetaData();
        Compute::Dense::Cuda::CreateCudnnConv2DMetaData(metaData, args...);
        std::lock_guard lock(m_cudnnConv2DMetaDataPoolMtx);
        m_cudnnConv2DMetaDataPool[convConfig] = metaData;
    }

    template <typename... Ts>
    static void AddCudnnPool2DMetaData(
        Compute::Dense::Cuda::PoolConfig poolConfig, Ts ... args)
    {
        auto* metaData = new Compute::Dense::Cuda::CudnnPool2DMetaData();
        Compute::Dense::Cuda::CreateCudnnPool2DMetaData(metaData, args...);
        std::lock_guard lock(m_cudnnPool2DMetaDataPoolMtx);
        m_cudnnPool2DMetaDataPool[poolConfig] = metaData;
    }

    static void AddCublasHandle(int deviceId, std::thread::id threadId);

    static void AddCudnnHandle(int deviceId, std::thread::id threadId);

    static void DeReferenceCuda(void* ptr, int deviceId);

    static void DeReferenceHost(void* ptr);

    static void ClearFreeCudaMemoryPool();

    static void ClearFreeHostMemoryPool();

    static void ClearCudaMemoryPool();

    static void ClearHostMemoryPool();

    static void ClearCudnnConv2DMetaDataPool();

    static void ClearCudnnPool2DMetaDataPool();

    static void ClearCublasHandlePool();

    static void ClearCudnnHandlePool();

    static void ClearBusyMemoryPool();

    static void ClearAll();

    static size_t GetTotalByteSizeCuda();

    static size_t GetTotalByteSizeHost();

    static size_t GetAllocatedByteSizeCuda();

    static size_t GetAllocatedByteSizeHost();

    static size_t GetFreeByteSizeCuda();

    static size_t GetFreeByteSizeHost();

    static bool HasConvConfig(Compute::Dense::Cuda::ConvConfig convConfig);

    static bool HasPoolConfig(Compute::Dense::Cuda::PoolConfig poolConfig);

    static bool HasCublasHandle(int deviceId, std::thread::id tid);

    static bool HasCudnnHandle(int deviceId, std::thread::id tid);

private:
    static std::unordered_multimap<size_t, MemoryChunk>
    m_hostFreeMemoryPool;
    static std::unordered_map<intptr_t, MemoryChunk> m_hostBusyMemoryPool;
    static std::unordered_multimap<std::size_t, MemoryChunk>
    m_cudaFreeMemoryPool;
    static std::unordered_map<intptr_t, MemoryChunk, std::hash<intptr_t>>
    m_cudaBusyMemoryPool;

    static std::unordered_map<intptr_t, void*> m_hostPreservedMemoryPool;
    static std::unordered_map<intptr_t, void*> m_cudaPreservedMemoryPool;

    static std::unordered_map<Compute::Dense::Cuda::ConvConfig,
                              Compute::Dense::Cuda::CudnnConv2DMetaData*,
                              ConvMetaDataHash>
    m_cudnnConv2DMetaDataPool;

    static std::unordered_map<Compute::Dense::Cuda::PoolConfig,
                              Compute::Dense::Cuda::CudnnPool2DMetaData*,
                              PoolMetaDataHash>
    m_cudnnPool2DMetaDataPool;

    //! Map for cublas and cudnn handles
    //! Key represents deviceId
    static std::unordered_map<std::pair<int, std::thread::id>, cublasHandle_t*,
                              DeviceIdTidHash>
    m_cublasHandlePool;
    static std::unordered_map<std::pair<int, std::thread::id>, cudnnHandle_t*,
                              DeviceIdTidHash>
    m_cudnnHandlePool;


    static std::mutex m_hostPoolMtx;
    static std::mutex m_cudaPoolMtx;
    static std::mutex m_cudnnConv2DMetaDataPoolMtx;
    static std::mutex m_cudnnPool2DMetaDataPoolMtx;
    static std::mutex m_cublasHandlePoolMtx;
    static std::mutex m_cudnnHandlePoolMtx;

    static unsigned int m_allocationUnitByteSize;
};
} // namespace Sapphire::Util

#endif  // Sapphire_MEMORYMANAGER_H
