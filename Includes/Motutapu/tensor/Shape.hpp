// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_SHAPE_DECL_HPP
#define MOTUTAPU_UTIL_SHAPE_DECL_HPP

#include <string>
#include <vector>

namespace Motutapu
{
class Shape
{
public:
    Shape() = default;
    ~Shape() = default;

    Shape(std::initializer_list<std::size_t> shape);
    Shape(std::vector<std::size_t> shape);

    Shape(const Shape& shape) = default;
    Shape(Shape&& shape) noexcept;

    Shape& operator=(const Shape& shape);
    Shape& operator=(Shape&& shape) noexcept;
    std::size_t& operator[](std::size_t index);

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;

    [[nodiscard]] std::string ToString() const;

    [[nodiscard]] std::size_t At(std::size_t index) const;

    [[nodiscard]] std::size_t Dim() const;

    [[nodiscard]] std::size_t Size() const noexcept;

private:
    std::vector<std::size_t> m_shapeVector;
};
} // namespace Takion

#endif
