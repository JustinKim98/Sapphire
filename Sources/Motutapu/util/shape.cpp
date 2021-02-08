// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/tensor/Shape.hpp>

namespace Motutapu
{
Shape::Shape(std::initializer_list<unsigned int> shape) : m_shapeVector(shape)
{
}

Shape::Shape(std::vector<unsigned int> shape) : m_shapeVector(std::move(shape))
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

unsigned int& Shape::operator[](unsigned int index)
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

unsigned int Shape::At(unsigned int index) const
{
    return m_shapeVector.at(index);
}

unsigned int Shape::Dim() const
{
    return static_cast<unsigned int>(m_shapeVector.size());
}

unsigned int Shape::Size() const noexcept
{
    unsigned int size = 1;
    for (auto i : m_shapeVector)
    {
        size *= i;
    }

    return size;
}

void Shape::Set(unsigned int dim, unsigned int value)
{
    if (dim >= m_shapeVector.size())
    {
        throw std::invalid_argument(
            "Shape::Set - Given dimension exceeds shape dimension");
    }

    m_shapeVector[dim] = value;
}

void Shape::Expand(unsigned int dim)
{
    std::vector<unsigned int> newShapeVector(dim);
    for (unsigned int i = 0; i < dim; i++)
    {
        if (i < dim - m_shapeVector.size())
            newShapeVector.at(i) = 1;
        else
            newShapeVector.at(i) =
                m_shapeVector.at(i - (dim - m_shapeVector.size()));
    }

    m_shapeVector = newShapeVector;
}

Shape Shape::GetTranspose() const
{
    if (m_shapeVector.size() < 2)
    {
        throw std::runtime_error(
            "GetTranspose - Shape must have dimension of at least 2 to perform "
            "transpose");
    }

    auto vector = m_shapeVector;
    auto temp = vector[vector.size() - 1];
    vector[vector.size() - 1] = vector[vector.size() - 2];
    vector[vector.size() - 2] = temp;

    return Shape(vector);
}

}  // namespace Motutapu
