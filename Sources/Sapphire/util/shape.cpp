// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/util/Shape.hpp>

namespace Sapphire
{
Shape::Shape(std::initializer_list<int> shape)
    : m_shapeVector(shape)
{
}

Shape::Shape(std::vector<int> shape)
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

int& Shape::operator[](int index)
{
    return m_shapeVector.at(index);
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

int Shape::At(int index) const
{
    return m_shapeVector.at(index);
}

int Shape::Dim() const
{
    return m_shapeVector.size();
}

int Shape::Size() const noexcept
{
    int size = 1;
    if (m_shapeVector.empty())
        return 0;
    for (auto i : m_shapeVector)
    {
        size *= i;
    }

    return size;
}

void Shape::Set(int dim, int value)
{
    if (dim >= static_cast<int>(m_shapeVector.size()))
    {
        throw std::invalid_argument(
            "Shape::Set - Given dimension exceeds shape dimension");
    }

    if (value == 0)
    {
        throw std::invalid_argument(
            "Shape::Set - Shape cannot have dimension with '0'");
    }

    m_shapeVector.at(dim) = value;
}

void Shape::SetRow(int value)
{
    if (m_shapeVector.size() < 2)
        throw std::runtime_error(
            "Shape::SetRow - Shape has less dimension than 2");

    m_shapeVector.at(static_cast<std::size_t>(Dim()) - 2) = value;
}

void Shape::SetCol(int value)
{
    if (m_shapeVector.empty())
        throw std::runtime_error("Shape::SetCol - Shape is empty");

    m_shapeVector.at(static_cast<std::size_t>(Dim()) - 1) = value;
}

void Shape::Expand(int dim)
{
    if (dim <= Dim())
        return;

    std::vector<int> newShapeVector(dim);
    for (int i = 0; i < dim; i++)
    {
        if (i < dim - static_cast<int>(m_shapeVector.size()))
            newShapeVector.at(i) = 1;
        else
            newShapeVector.at(i) =
                m_shapeVector.at(i - (dim - m_shapeVector.size()));
    }

    m_shapeVector = newShapeVector;
}

void Shape::Squeeze(int dim)
{
    if (dim >= Dim())
        return;

    const auto dimIdx = m_shapeVector.size() - 1 - dim;

    if (m_shapeVector.at(dimIdx) > 1)
        return;

    std::vector<int> newShapeVector(m_shapeVector.size() - 1);
    int newIdx = static_cast<int>(m_shapeVector.size()) - 2;

    for (int i = static_cast<int>(m_shapeVector.size()) - 1; i >= 0; --i)
        if (i != static_cast<int>(dimIdx))
        {
            newShapeVector.at(newIdx) = m_shapeVector.at(i);
            newIdx -= 1;
        }
}

void Shape::Squeeze()
{
    std::vector<int> newShapeVector;
    newShapeVector.reserve(m_shapeVector.size());

    for (int i : m_shapeVector)
    {
        if (i > 1)
            newShapeVector.emplace_back(i);
    }

    m_shapeVector = newShapeVector;
}

void Shape::Shrink(int dim)
{
    if (dim >= Dim())
        return;

    const auto dimIdx = m_shapeVector.size() - dim;
    std::vector<int> newShapeVector(dim);

    for (int i = static_cast<int>(m_shapeVector.size()) - 1; i >= 0; --i)
    {
        if (i >= static_cast<int>(dimIdx))
            newShapeVector.at(i - (m_shapeVector.size() - dim)) =
                m_shapeVector.at(i);
        else
            newShapeVector.at(0) *= m_shapeVector.at(i);
    }
}

int Shape::GetBatchSize(int requiredDim) const
{
    if (const auto dim = Dim(); dim > requiredDim)
    {
        int batchSize = 1;
        for (int i = 0; i < dim - requiredDim; ++i)
        {
            batchSize *= At(i);
        }
        return batchSize;
    }
    return 1;
}

Shape Shape::GetTranspose() const
{
    if (m_shapeVector.empty())
    {
        throw std::runtime_error(
            "GetTranspose - Shape cannot be empty  to perform "
            "transpose");
    }

    if (m_shapeVector.size() == 1)
    {
        std::vector<int> newShape(2);
        newShape.at(0) = m_shapeVector.at(0);
        newShape.at(1) = 1;
        return Shape(newShape);
    }

    auto vector = m_shapeVector;
    const auto temp = vector.at(vector.size() - 1);
    vector.at(vector.size() - 1) = vector.at(vector.size() - 2);
    vector.at(vector.size() - 2) = temp;

    return Shape(vector);
}
} // namespace Sapphire
