// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_SPINLOCK_HPP
#define SAPPHIRE_UTIL_SPINLOCK_HPP
#include <atomic>
namespace Sapphire::Util
{
class SpinLock
{
 public:
    static void Lock(std::atomic<bool>* ptr)
    {
        while (true)
        {
            if (!(*ptr).exchange(true, std::memory_order_acquire))
                break;
            //! Loop on load in order to reduce cache coherency traffic
            while ((*ptr).load(std::memory_order_relaxed))
            {
                //! Pause instructions are used in order to prevent starving of
                //! other cores
#if defined(__GNUC__)
                __builtin_ia32_pause();
#elif defined(_MSC_VER)
                _mm_pause();
#endif
            }
        }
    }

    static void Release(std::atomic<bool>* ptr)
    {
        (*ptr).store(false, std::memory_order_release);
    }
};
}  // namespace Sapphire::Util

#endif  // SAPPHIRE_SPINLOCK_HPP
