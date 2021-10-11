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
#include <Sapphire/util/HashFunctions.hpp>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace Sapphire::Util
{
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
    static void* GetMemoryCuda(size_t byteSize, bool preserve = false);

    //! Allocates memory on host
    //! \param byteSize : Allocation size in bytes
    static void* GetMemoryHost(size_t byteSize, bool preserve = false);

    static void FreePreservedHost(void* ptr);

    static void FreePreservedCuda(void* ptr);

    static void MoveToPreservedHost(void* ptr);

    static void MoveToPreservedCuda(void* ptr);

    static void MoveToVolatileHost(void* ptr);

    static void MoveToVolatileCuda(void* ptr);

    static Compute::Dense::Cuda::CudnnConv2DMetaData* GetCudnnConvMetaData(
        Compute::Dense::Cuda::ConvConfig convConfig);

    static Compute::Dense::Cuda::CudnnPool2DMetaData* GetCudnnPoolMetaData(
        Compute::Dense::Cuda::PoolConfig poolConfig);

    static cublasHandle_t* GetCublasHandle(int deviceId,
                                           std::thread::id threadId);

    static cudnnHandle_t*
    GetCudnnHandle(int deviceId, std::thread::id threadId);

    template <typename ...Ts>
    static void AddCudnnConv2DMetaData(
        Compute::Dense::Cuda::ConvConfig convConfig, Ts ... args)
    {
        auto* metaData = new Compute::Dense::Cuda::CudnnConv2DMetaData();
        Compute::Dense::Cuda::CreateCudnnConv2DMetaData(metaData, args...);
        m_cudnnConv2DMetaDataPool[convConfig] = metaData;
    }

    template <typename... Ts>
    static void AddCudnnPool2DMetaData(
        Compute::Dense::Cuda::PoolConfig poolConfig, Ts ... args)
    {
        auto* metaData = new Compute::Dense::Cuda::CudnnPool2DMetaData();
        Compute::Dense::Cuda::CreateCudnnPool2DMetaData(metaData, args...);
        m_cudnnPool2DMetaDataPool[poolConfig] = metaData;
    }

    static void AddCublasHandle(int deviceId, std::thread::id threadId);

    static void AddCudnnHandle(int deviceId, std::thread::id threadId);

    static void ClearCudnnConv2DMetaDataPool();

    static void ClearCudnnPool2DMetaDataPool();

    static void ClearCublasHandlePool();

    static void ClearCudnnHandlePool();

    static void Clean();

    static void ClearFreeMemoryPool();

    static void ClearPreservedMemoryPool();

    static void ClearAll();

    static std::size_t GetTotalByteSizeCuda();

    static std::size_t GetTotalByteSizeHost();

    static bool HasConvConfig(Compute::Dense::Cuda::ConvConfig convConfig);

    static bool HasPoolConfig(Compute::Dense::Cuda::PoolConfig poolConfig);

    static bool HasCublasHandle(int deviceId, std::thread::id tid);

    static bool HasCudnnHandle(int deviceId, std::thread::id tid);

private:
    //! Memory resources
    static std::unordered_multimap<std::size_t, MemoryChunk>
    m_hostFreeMemoryPool;
    static std::unordered_map<std::intptr_t, MemoryChunk> m_hostBusyMemoryPool;
    static std::unordered_multimap<std::size_t, MemoryChunk>
    m_cudaFreeMemoryPool;
    static std::unordered_map<std::intptr_t, MemoryChunk>
    m_cudaBusyMemoryPool;

    static std::unordered_map<std::intptr_t, MemoryChunk>
    m_hostPreservedMemoryPool;
    static std::unordered_map<std::intptr_t, MemoryChunk>
    m_cudaPreservedMemoryPool;

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

    static unsigned int m_allocationUnitByteSize;
};
} // namespace Sapphire::Util

#endif  // Sapphire_MEMORYMANAGER_H
