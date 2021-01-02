// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/util/ShapeDecl.hpp>

namespace Motutapu::Util
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

//Shape Shape::operator*(const Shape& shape) const
//{
//    
//}
//
//void Shape::Expand(std::size_t rank)
//{
//    
//}


} // namespace Motutapu::Util
