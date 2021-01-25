// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_SHAPE_DECL_HPP
#define MOTUTAPU_UTIL_SHAPE_DECL_HPP

#include <string>
#include <vector>

namespace Motutapu
{
enum class Type
{
    Sparse,
    Dense,
};

class Shape
{
public:
    Shape() = default;
    ~Shape() = default;

    Shape(std::initializer_list<unsigned int> shape);
    Shape(std::vector<unsigned int> shape);

    Shape(const Shape& shape) = default;
    Shape(Shape&& shape) noexcept;

    Shape& operator=(const Shape& shape);
    Shape& operator=(Shape&& shape) noexcept;
    unsigned int& operator[](unsigned int index);

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;

    [[nodiscard]] std::string ToString() const;

    [[nodiscard]] unsigned int At(unsigned int index) const;

    [[nodiscard]] unsigned int Dim() const;

    [[nodiscard]] unsigned int Size() const noexcept;

private:
    std::vector<unsigned int> m_shapeVector;
};
} // namespace Takion

#endif
