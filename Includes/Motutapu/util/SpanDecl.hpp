// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_SPAN_DECL_HPP
#define MOTUTAPU_SPAN_DECL_HPP

#include <cstring>
#include <iterator>
#include <stdexcept>

namespace Motutapu::Util
{
template <typename T>
class Span
{
 public:
    Span() noexcept : m_base(nullptr), m_length(0)
    {
    }

    Span(T* base, std::size_t length) noexcept : m_base(base), m_length(length)
    {
    }

    template <typename It>
    Span(It begin, It end)
        : m_base(&*begin),
          m_length(static_cast<std::size_t>(std::distance(begin, end)))
    {
    }

    ~Span() = default;

    Span(const Span& span) = default;

    Span(Span&& span) noexcept = default;

    Span& operator=(const Span& span) = default;

    Span& operator=(Span&& span) noexcept = default;

    const T* Base() const
    {
        return m_base;
    }

    T* Address(std::size_t idx)
    {
        return m_base + idx;
    }

    const T* Address(std::size_t idx) const
    {
        return m_base + idx;
    }

    T& operator[](std::size_t idx)
    {
        return m_base[idx];
    }

    const T& operator[](std::size_t idx) const
    {
        return m_base[idx];
    }

    T& At(std::size_t idx)
    {
        if (idx > m_length)
            throw std::invalid_argument("index exceeds original span");
        return m_base[idx];
    }

    const T& At(std::size_t idx) const
    {
        if (idx > m_length)
            throw std::invalid_argument("index exceeds original span");
        return m_base[idx];
    }

    T* Begin()
    {
        return m_base;
    }

    T* End()
    {
        return m_base + m_length;
    }

    std::size_t Length() const
    {
        return m_length;
    }

    Span<T> SubSpan(std::size_t offset, std::size_t length)
    {
        if (offset + length > m_length)
            throw std::invalid_argument("Subspan index exceeds original span");
        return Span<T>(m_base + offset, length);
    }

    void Clear()
    {
        if (m_base != nullptr)
        {
#ifdef _MSC_VER
            _aligned_free(m_base);
#else
            free(m_base);
#endif
        }
        m_base = nullptr;
        m_length = 0;
    }

    static void DeepCopy(Span<T> dst, Span<T> src)
    {
        if (dst.m_length != src.m_length)
            std::invalid_argument(
                "Data Length mismatch between source and destination");
        std::memcpy(dst.m_base, src.m_base, src.m_length * sizeof(T));
    }

 private:
    T* m_base;
    std::size_t m_length;
};
}  // namespace MOTUTAPU::Util

#endif
