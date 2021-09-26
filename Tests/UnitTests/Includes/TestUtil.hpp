// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TESTS_TEST_UTIL_HPP
#define SAPPHIRE_TESTS_TEST_UTIL_HPP

#define FP_EQUAL_THRESHOLD  0.001

#include <type_traits>
#include <limits>

namespace Sapphire::Test
{
template <typename T, std::enable_if_t<std::is_integral_v<T>, bool>  = true>
inline bool TestEquality(T a, T b)
{
    return a == b;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool>  =
              true>
inline bool TestEquality(T a, T b)
{
    return std::abs(a - b) < static_cast<T>(FP_EQUAL_THRESHOLD);
}
}
#endif
