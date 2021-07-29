// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_SPINLOCK_HPP
#define SAPPHIRE_UTIL_SPINLOCK_HPP
#include <atomic>
#include <emmintrin.h>

namespace Sapphire::Util
{
class SpinMutex
{
public:
    SpinMutex() = default;

    void Lock()
    {
        while (!m_flag.test_and_set(std::memory_order_acquire))
#if defined(__GNUC__)
            __builtin_ia32_pause();
#elif defined(_MSC_VER)
            _mm_pause();
#endif
    }

    void Release()
    {
        m_flag.clear(std::memory_order_release);
    }

private:
    std::atomic_flag m_flag = ATOMIC_FLAG_INIT;
};
} // namespace Sapphire::Util

//! TODO : implement shared spin-lock

#endif  // SAPPHIRE_SPINLOCK_HPP
