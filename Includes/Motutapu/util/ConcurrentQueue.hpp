// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CONCURRENTQUEUE_HPP
#define MOTUTAPU_CONCURRENTQUEUE_HPP

#include <optional>
#include <shared_mutex>
#include <vector>

namespace Motutapu::Util
{
template <typename T>
class ConcurrentQueue
{
public:
    ConcurrentQueue(std::size_t maxSize)
        : m_vector(maxSize)
    {
    }

    template <typename U>
    void TryPush(U&& object)
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value);
        //! This lock will be released automatically if function returns (same
        //! for all lock_guards used in this code)
        std::lock_guard lock(m_mtx);

        // Return if capacity is full (Waits if given predicate (second
        // argument) is false)
        if (Size() == m_vector.size())
            return;

        if (m_startIdx == m_endIdx)
        {
            m_PopCondVar.notify_one();
        }

        m_endIdx = m_endIdx == m_vector.size() ? 0 : m_endIdx + 1;

        const long elemIdx = m_endIdx - 1 < 0 ? m_vector.size() : m_endIdx - 1;

        m_vector[static_cast<std::size_t>(elemIdx)] = std::forward<U>(object);
    }

    //! Pushes element into the queue
    //! Waits if element is full
    template <typename U>
    void Push(U&& object)
    {
        static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value);
        std::lock_guard lock(m_mtx);

        // Wait if capacity is full (Waits if given predicate (second argument)
        // is false)
        m_PushCondVar.wait(m_mtx,
                           [this]() { return Size() != m_vector.size(); });

        if (m_startIdx == m_endIdx)
        {
            m_PopCondVar.notify_one();
        }

        m_endIdx = m_endIdx == m_vector.size() ? 0 : m_endIdx + 1;

        const long elemIdx = m_endIdx - 1 < 0 ? m_vector.size() : m_endIdx - 1;

        m_vector[static_cast<std::size_t>(elemIdx)] = std::forward<U>(object);
    }

    //! Tries Push element into the queue
    //! returns immediately if element is full
    //! \return : optional value of element. Popped element if successful,
    //! std::nullopt if failure
    std::optional<T> TryPop()
    {
        std::lock_guard lock(m_mtx);

        //!  if Queue is empty (Waits if given predicate (second argument)
        //! is false)
        if (m_startIdx == m_endIdx)
        {
            return {};
        }

        if (Size() == m_vector.size())
        {
            m_PushCondVar.notify_one();
        }

        m_startIdx = m_startIdx == m_vector.size() - 1 ? 0 : m_startIdx + 1;

        return m_vector[m_startIdx];
    }

    //! Pops element from the queue
    //! If queue is empty, it waits until element is inserted by Push
    //! \return : Popped element
    T Pop()
    {
        std::lock_guard lock(m_mtx);

        //! Wait if Queue is empty (Waits if given predicate (second argument)
        //! is false)
        m_PopCondVar.wait(m_mtx, [this]() { return m_startIdx != m_endIdx; });

        if (Size() == m_vector.size())
        {
            m_PushCondVar.notify_one();
        }

        m_startIdx = m_startIdx == m_vector.size() - 1 ? 0 : m_startIdx + 1;

        return m_vector[m_startIdx];
    }

    //! Invokes given Handler with parameters if queue is not empty
    //! Returns immediately if queue is empty
    //! \tparam Func : Type for the handler function
    //! \tparam Ts : Additional parameters for the handler (If required)
    template <typename Func, typename... Ts>
    void TryInvoke(Func handler, Ts ... params)
    {
        auto elem = TryPop();
        if (elem)
        {
            handler(elem.value(), params...);
        }
    }

    //! Invokes given Handler with parameters
    //! Waits if queue is empty
    //! \tparam Func : Type for the handler function
    //! \tparam Ts : Additional parameters for the handler (If required)
    template <typename Func, typename... Ts>
    void Invoke(Func handler, Ts ... params)
    {
        handler(Pop(), params...);
    }

    //! Returns size of the ConcurrentQueue
    //! If m_endIdx indicates index for end of the queue (last element stored in
    //! m_endIdx - 1) while m_startIdx indicates index for start of the queue.
    //! Therefore, m_startIdx == m_endIdx indicates empty queue
    std::size_t Size()
    {
        if (m_endIdx < m_startIdx)
            return (m_vector.size() - m_startIdx) + m_endIdx;
        return m_endIdx - m_startIdx;
    }

private:
    std::vector<T> m_vector;
    long m_startIdx = 0;
    long m_endIdx = 0;
    std::shared_mutex m_mtx;
    std::condition_variable m_PushCondVar;
    std::condition_variable m_PopCondVar;
};
}

#endif
