// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_SHARED_PTR_HPP
#define SAPPHIRE_UTIL_SHARED_PTR_HPP

#include <Sapphire/util/SharedPtrDecl.hpp>
#include <Sapphire/util/Spinlock.hpp>

namespace Sapphire::Util
{
template <typename T>
void SharedPtr<T>::m_delete() const
{
    if (m_sharedObjectInfoPtr)
    {
        SpinLock::Lock(&m_sharedObjectInfoPtr->Busy);
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
            SpinLock::Release(&m_sharedObjectInfoPtr->Busy);
        }
    }
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(U* objectPtr,
                        SharedObjectInfo* informationPtr)
    : m_id(0),
      m_objectPtr(objectPtr),
      m_sharedObjectInfoPtr(informationPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(
    const SharedPtr<U>& sharedPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    SpinLock::Lock(&sharedPtr.m_sharedObjectInfoPtr->Busy);
    sharedPtr.m_sharedObjectInfoPtr->RefCount.fetch_add(
        1, std::memory_order_relaxed);
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    SpinLock::Release(&sharedPtr.m_sharedObjectInfoPtr->Busy);
}

template <typename T>
SharedPtr<T>::SharedPtr(const SharedPtr<T>& sharedPtr)
{
    SpinLock::Lock(&sharedPtr.m_sharedObjectInfoPtr->Busy);
    sharedPtr.m_sharedObjectInfoPtr->RefCount.fetch_add(
        1, std::memory_order_relaxed);
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    SpinLock::Release(&sharedPtr.m_sharedObjectInfoPtr->Busy);
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(
    SharedPtr<U>&& sharedPtr) noexcept
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    auto* objectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    SpinLock::Lock(&objectInfoPtr->Busy);
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;
    SpinLock::Release(&objectInfoPtr->Busy);
}

template <typename T>
SharedPtr<T>::SharedPtr(
    SharedPtr<T>&& sharedPtr) noexcept
{
    auto* objectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    SpinLock::Lock(&objectInfoPtr->Busy);
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;
    SpinLock::Release(&objectInfoPtr->Busy);
}

template <typename T>
template <typename U>
SharedPtr<T>& SharedPtr<T>::operator=(
    const SharedPtr<U>& sharedPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    if (this == &sharedPtr)
        return *this;

    m_delete();
    auto* objectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    SpinLock::Lock(&objectInfoPtr->Busy);
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    sharedPtr.m_sharedObjectInfoPtr->RefCount.fetch_add(1);
    SpinLock::Release(&objectInfoPtr->Busy);
    return *this;
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(
    const SharedPtr<T>& sharedPtr)
{
    if (this == &sharedPtr)
        return *this;

    m_delete();
    auto* objectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    SpinLock::Lock(&objectInfoPtr->Busy);
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    sharedPtr.m_sharedObjectInfoPtr->RefCount.fetch_add(1);
    SpinLock::Release(&objectInfoPtr->Busy);
    return *this;
}

template <typename T>
template <typename U>
SharedPtr<T>& SharedPtr<T>::operator=(
    SharedPtr<U>&& sharedPtr)
noexcept
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    m_delete();
    auto* sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    SpinLock::Lock(&sharedObjectInfoPtr->Busy);
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;
    sharedObjectInfoPtr->RefCount.fetch_add(1);
    SpinLock::Release(&sharedObjectInfoPtr->Busy);
    return *this;
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(
    SharedPtr<T>&& sharedPtr) noexcept
{
    m_delete();
    auto* sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    SpinLock::Lock(&sharedObjectInfoPtr->Busy);
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;
    sharedObjectInfoPtr->RefCount.fetch_add(1);
    SpinLock::Release(&sharedObjectInfoPtr->Busy);
    return *this;
}


template <typename T>
template <typename U>
bool SharedPtr<T>::operator==(
    const SharedPtr<U>& sharedPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    return m_sharedObjectInfoPtr == sharedPtr.m_sharedObjectInfoPtr &&
           m_objectPtr == sharedPtr.m_objectPtr;
}

template <typename T>
bool SharedPtr<T>::operator==(const SharedPtr<T>& sharedPtr)
{
    auto equality = m_sharedObjectInfoPtr == sharedPtr.m_sharedObjectInfoPtr &&
                    m_objectPtr == sharedPtr.m_objectPtr;
    return equality;
}

template <typename T>
template <typename U>
bool SharedPtr<T>::operator!=(
    const SharedPtr<U>& sharedPtr)
{
    return !(*this == sharedPtr);
}

template <typename T>
bool SharedPtr<T>::operator!=(
    const SharedPtr<T>& sharedPtr)
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
} // namespace Sapphire::Util

#endif  // Takion_SHAREDPTR_IMPL_HPP
