// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_SHAPE_DECL_HPP
#define MOTUTAPU_UTIL_SHAPE_DECL_HPP

#include <stdexcept>
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
    explicit Shape(std::vector<unsigned int> shape);

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

    [[nodiscard]] std::vector<unsigned int> GetShapeVector() const
    {
        return m_shapeVector;
    }

    void Set(unsigned int dim, unsigned int value);

    [[nodiscard]] unsigned int Rows() const
    {
        return m_shapeVector.size() > 1
                   ? m_shapeVector.at(m_shapeVector.size() - 2)
                   : 1;
    }

    [[nodiscard]] unsigned int Cols() const
    {
        return !m_shapeVector.empty()
                   ? m_shapeVector.at(m_shapeVector.size() - 1)
                   : 0;
    }

    //! Expands the shape to dim
    //! If shape has already equal or higher dimension than requested dimension,
    //! returns immediately
    void Expand(unsigned int dim);

    Shape GetReverse() const;

    [[nodiscard]] Shape GetTranspose() const;

 private:
    std::vector<unsigned int> m_shapeVector;
};
}  // namespace Motutapu

#endif
