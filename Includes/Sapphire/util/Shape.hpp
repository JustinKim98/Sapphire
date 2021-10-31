// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_SHAPE_DECL_HPP
#define SAPPHIRE_UTIL_SHAPE_DECL_HPP

#include <stdexcept>
#include <string>
#include <vector>

namespace Sapphire
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

    Shape(std::initializer_list<int> shape);
    explicit Shape(std::vector<int> shape);

    Shape(const Shape& shape) = default;
    Shape(Shape&& shape) noexcept;

    Shape& operator=(const Shape& shape);
    Shape& operator=(Shape&& shape) noexcept;
    int& operator[](int index);

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;

    [[nodiscard]] std::string ToString() const;

    [[nodiscard]] int At(int index) const;

    //! Get the total dimension
    [[nodiscard]] int Dim() const;

    //! Get number of total elements
    [[nodiscard]] int Size() const noexcept;

    [[nodiscard]] std::vector<int> GetShapeVector() const
    {
        return m_shapeVector;
    }

    void Set(int index, int value);

    [[nodiscard]] int Rows() const
    {
        return m_shapeVector.size() > 1
                   ? m_shapeVector.at(m_shapeVector.size() - 2)
                   : 1;
    }

    [[nodiscard]] int Cols() const
    {
        return !m_shapeVector.empty()
                   ? m_shapeVector.at(m_shapeVector.size() - 1)
                   : 0;
    }

    //! Expands the shape to dim
    //! If shape has already equal or higher dimension than requested dimension,
    //! returns immediately
    void Expand(int dim);

    //! Removes the dimension if given dimension has size 1
    void Squeeze(int index);

    //! Removes all 1's in the given shape
    void Squeeze();

    //! Shrinks dimension to given dim
    void Shrink(int dim);

    [[nodiscard]] int GetNumUnits(int requiredDim) const;

    [[nodiscard]] int GetUnitSize(int requiredDim) const;


    Shape GetReverse() const;

    [[nodiscard]] Shape GetTranspose() const;

private:
    std::vector<int> m_shapeVector;
};
} // namespace Sapphire

#endif
