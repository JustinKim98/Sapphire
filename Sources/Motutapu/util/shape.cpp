// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/tensor/Shape.hpp>

namespace Motutapu
{
Shape::Shape(std::initializer_list<std::size_t> shape)
    : m_shapeVector(shape)
{
}

Shape::Shape(std::vector<std::size_t> shape)
    : m_shapeVector(std::move(shape))
{
}

Shape::Shape(Shape&& shape) noexcept
    : m_shapeVector(std::move(shape.m_shapeVector))
{
}

Shape& Shape::operator=(const Shape& shape)
{
    if (this == &shape)
        return *this;
    m_shapeVector = shape.m_shapeVector;
    return *this;
}

Shape& Shape::operator=(Shape&& shape) noexcept
{
    m_shapeVector = std::move(shape.m_shapeVector);
    return *this;
}

std::size_t& Shape::operator[](std::size_t index)
{
    return m_shapeVector[index];
}

bool Shape::operator==(const Shape& shape) const
{
    return m_shapeVector == shape.m_shapeVector;
}


bool Shape::operator!=(const Shape& shape) const
{
    return m_shapeVector != shape.m_shapeVector;
}

std::string Shape::ToString() const
{
    std::string msg;
    msg += "Dim : " + std::to_string(Dim()) + " ";
    msg += " [";

    for (auto dim : m_shapeVector)
        msg += (std::to_string(dim) + " ");

    msg += " ] ";
    return msg;
}

std::size_t Shape::At(std::size_t index) const
{
    return m_shapeVector.at(index);
}

std::size_t Shape::Dim() const
{
    return m_shapeVector.size();
}

std::size_t Shape::Size() const noexcept
{
    std::size_t size = 1;
    for (auto i : m_shapeVector)
    {
        size *= i;
    }

    return size;
}

} // namespace Motutapu::Util
