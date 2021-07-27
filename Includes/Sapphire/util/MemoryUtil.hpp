// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_MEMORY_UTIL_HPP
#define SAPPHIRE_UTIL_MEMORY_UTIL_HPP
# define MEM_ALIGN 64

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <Sapphire/util/ResourceManager.hpp>
#endif
#include <type_traits>

namespace Sapphire::Util
{
template <typename T>
class AlignedDelete
{
    constexpr AlignedDelete() noexcept = default;

    template <typename U>
    AlignedDelete(const AlignedDelete<U>&) noexcept
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                      std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    }

    void operator()(T* ptr) const
    {
#ifdef _MSC_VER
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
};

template <typename T>
class HostDeleteManaged
{
    constexpr HostDeleteManaged() noexcept = default;

    template <typename U>
    HostDeleteManaged(const HostDeleteManaged<U>&)
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                      std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    }

    void operator()(T* ptr)
    {
        ResourceManager::DeReferenceHost(static_cast<void*>(ptr));
    }
};

#ifdef USE_CUDA
template <typename T>
class CudaDelete
{
    constexpr CudaDelete() noexcept = default;

    template <typename U>
    CudaDelete(const CudaDelete<U>&)
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                      std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    }

    void operator()(T* ptr)
    {
        cudaFree(ptr);
    }
};

template <typename T>
class CudaDeleteManaged
{
    constexpr CudaDeleteManaged() noexcept = default;

    template <typename U>
    CudaDeleteManaged(const CudaDeleteManaged<U>&)
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                      std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    }

    void operator()(T* ptr, int deviceId)
    {
        ResourceManager::DeReferenceCuda(static_cast<void*>(ptr),
                                         deviceId);
    }
};
#endif

template <typename T>
class DefaultAllocate
{
    constexpr DefaultAllocate() noexcept = default;

    template <typename U>
    DefaultAllocate(const DefaultAllocate<U>&) noexcept
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                      std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    }

    void operator()(T** ptr, std::size_t size) const
    {
        *ptr = new T[size];
    }
};


template <typename T>
class AlignedAllocate
{
    constexpr AlignedAllocate() noexcept = default;

    template <typename U>
    AlignedAllocate(const AlignedAllocate<U>&) noexcept
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                      std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    }

    void operator()(T** ptr, std::size_t size) const
    {
#ifdef _MSC_VER
        *ptr = static_cast<T*>(_aligned_malloc(size * sizeof(T), MEM_ALIGN));
#else
        *ptr = static_cast<T*>(aligned_alloc(MEM_ALIGN, size * sizeof(T)));
#endif
    }
};

template <typename T>
class CudaAllocate
{
    constexpr CudaAllocate() noexcept = default;

    template <typename U>
    CudaAllocate(const CudaAllocate<U>&) noexcept
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                      std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    }

    void operator()(T** ptr, std::size_t size) const
    {
        cudaMalloc(static_cast<void**>(*ptr), size * sizeof(T));
    }
};

template <typename T>
class HostAllocateManaged
{
    constexpr HostAllocateManaged() noexcept = default;

    template <typename U>
    HostAllocateManaged(const HostAllocateManaged<U>&) noexcept
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                      std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    }

    void operator()(T** ptr, std::size_t size) const
    {
        *ptr =
            static_cast<T*>(ResourceManager::GetMemoryHost(size * sizeof(T)));
    }
};

template <typename T>
class CudaAllocateManaged
{
    constexpr CudaAllocateManaged() noexcept = default;

    template <typename U>
    CudaAllocateManaged(const CudaAllocateManaged<U>&) noexcept
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                      std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    }

    void operator()(T** ptr, std::size_t size, int deviceId) const
    {
        *ptr = static_cast<T*>(
            ResourceManager::GetMemoryCuda(size * sizeof(T), deviceId));
    }
};
}

#endif
