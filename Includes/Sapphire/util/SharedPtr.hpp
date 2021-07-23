// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_SHAREDPTR_HPP
#define SAPPHIRE_SHAREDPTR_HPP

#include <Sapphire/util/SharedPtrDecl.hpp>

namespace Sapphire::Util
{
template <typename T>
void SharedPtr<T>::m_delete() const
{
    if (m_sharedObjectInfoPtr)
    {
        if (m_sharedObjectInfoPtr->RefCount.load(std::memory_order_acquire) ==
            1)
        {
            delete m_sharedObjectInfoPtr;
            delete m_objectPtr;
        }
        else
        {
            m_sharedObjectInfoPtr->RefCount.fetch_sub(
                1, std::memory_order_release);
        }
    }
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(U* objectPtr, SharedObjectInfo* informationPtr)
    : m_id(0), m_objectPtr(objectPtr), m_sharedObjectInfoPtr(informationPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    ;
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(const SharedPtr<U>& sharedPtr)
    : m_objectPtr(sharedPtr.m_objectPtr),
      m_sharedObjectInfoPtr(sharedPtr.m_sharedObjectInfoPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    if (sharedPtr.m_sharedObjectInfoPtr)
    {
        int oldRefCount = sharedPtr.m_sharedObjectInfoPtr->RefCount.load(
            std::memory_order_relaxed);
        while (!sharedPtr.m_sharedObjectInfoPtr->RefCount.compare_exchange_weak(
            oldRefCount, oldRefCount + 1, std::memory_order_release,
            std::memory_order_relaxed))
            ;
    }
}

template <typename T>
SharedPtr<T>::SharedPtr(const SharedPtr<T>& sharedPtr)
{
    if (sharedPtr.m_sharedObjectInfoPtr)
    {
        int oldRefCount = sharedPtr.m_sharedObjectInfoPtr->RefCount.load(
            std::memory_order_relaxed);
        while (!sharedPtr.m_sharedObjectInfoPtr->RefCount.compare_exchange_weak(
            oldRefCount, oldRefCount + 1, std::memory_order_release,
            std::memory_order_relaxed))
            ;
    }
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(SharedPtr<U>&& sharedPtr) noexcept
    : m_objectPtr(sharedPtr.m_objectPtr),
      m_sharedObjectInfoPtr(sharedPtr.m_sharedObjectInfoPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;
}

template <typename T>
SharedPtr<T>::SharedPtr(SharedPtr<T>&& sharedPtr) noexcept
{
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;
}

template <typename T>
template <typename U>
SharedPtr<T>& SharedPtr<T>::operator=(const SharedPtr<U>& sharedPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    if (this == &sharedPtr)
        return *this;

    int oldRefCount = sharedPtr.m_sharedObjectInfoPtr->RefCount.load(
        std::memory_order_relaxed);
    while (!sharedPtr.m_sharedObjectInfoPtr->RefCount.compare_exchange_weak(
        oldRefCount, oldRefCount + 1, std::memory_order_release,
        std::memory_order_relaxed))
        ;

    m_delete();
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;

    return *this;
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(const SharedPtr<T>& sharedPtr)
{
    if (this == &sharedPtr)
        return *this;

    int oldRefCount = sharedPtr.m_sharedObjectInfoPtr->RefCount.load(
        std::memory_order_relaxed);
    while (!sharedPtr.m_sharedObjectInfoPtr->RefCount.compare_exchange_weak(
        oldRefCount, oldRefCount + 1, std::memory_order_release,
        std::memory_order_relaxed))
        ;

    m_delete();
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;

    return *this;
}

template <typename T>
template <typename U>
SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr<U>&& sharedPtr) noexcept
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    if (*this != sharedPtr)
    {
        m_delete();
        m_objectPtr = sharedPtr.m_objectPtr;
        m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    }

    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;

    return *this;
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr<T>&& sharedPtr) noexcept
{
    if (*this != sharedPtr)
    {
        m_delete();
        m_objectPtr = sharedPtr.m_objectPtr;
        m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    }

    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;

    return *this;
}

template <typename T>
template <typename U>
bool SharedPtr<T>::operator==(const SharedPtr<U>& sharedPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    return m_sharedObjectInfoPtr == sharedPtr.m_sharedObjectInfoPtr &&
           m_objectPtr == sharedPtr.m_objectPtr;
}

template <typename T>
bool SharedPtr<T>::operator==(const SharedPtr<T>& sharedPtr)
{
    return m_sharedObjectInfoPtr == sharedPtr.m_sharedObjectInfoPtr &&
           m_objectPtr == sharedPtr.m_objectPtr;
}

template <typename T>
template <typename U>
bool SharedPtr<T>::operator!=(const SharedPtr<U>& sharedPtr)
{
    return !(*this == sharedPtr);
}

template <typename T>
bool SharedPtr<T>::operator!=(const SharedPtr<T>& sharedPtr)
{
    return !(*this == sharedPtr);
}

template <typename T>
SharedPtr<T>::~SharedPtr()
{
    m_delete();
}

template <typename T>
auto& SharedPtr<T>::operator[](std::ptrdiff_t idx)
{
    return m_objectPtr[idx];
}

template <typename T>
SharedPtr<T> SharedPtr<T>::Make(T* objectPtr)
{
    auto* infoPtr = new SharedObjectInfo();
    return SharedPtr<T>(objectPtr, infoPtr);
}

template <typename T>
SharedPtr<T> SharedPtr<T>::Make()
{
    T* objectPtr = new T();
    auto* infoPtr = new SharedObjectInfo();
    return SharedPtr<T>(objectPtr, infoPtr);
}

template <typename T>
template <typename... Ts>
SharedPtr<T> SharedPtr<T>::Make(Ts&&... args)
{
    T* objectPtr = new T(std::forward<Ts>(args)...);
    auto* infoPtr = new SharedObjectInfo();
    return SharedPtr<T>(objectPtr, infoPtr);
}

template <typename T>
T* SharedPtr<T>::operator->() const
{
    return m_objectPtr;
}
}  // namespace Sapphire::Util

#endif  // Takion_SHAREDPTR_IMPL_HPP