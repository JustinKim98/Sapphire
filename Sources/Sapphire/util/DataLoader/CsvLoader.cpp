// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/util/DataLoader/DataLoader.hpp>
#include <Sapphire/util/DataLoader/CsvLoader.hpp>
#include <iostream>
#include <fstream>

namespace Sapphire::Util
{
std::string_view CSVRow::operator[](std::size_t index) const
{
    return std::string_view(&m_line[m_data[index] + 1],
                            m_data[index + 1] - (m_data[index] + 1));
}

std::size_t CSVRow::size() const
{
    return m_data.size() - 1;
}

void CSVRow::ReadNextRow(std::istream& str)
{
    std::getline(str, m_line);

    m_data.clear();
    m_data.emplace_back(-1);
    std::string::size_type pos = 0;
    while ((pos = m_line.find(',', pos)) != std::string::npos)
    {
        m_data.emplace_back(pos);
        ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos = m_line.size();
    m_data.emplace_back(pos);
}

inline std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.ReadNextRow(str);
    return str;
}

CSVIterator::CSVIterator(std::istream& str)
    : m_str(str.good() ? &str : nullptr)
{
    ++*this;
}

CSVIterator& CSVIterator::operator++()
{
    if (m_str)
    {
        if (!(*m_str >> m_row))
        {
            m_str = nullptr;
        }
    }
    return *this;
}

CSVIterator CSVIterator::operator++(int)
{
    CSVIterator tmp(*this);
    ++(*this);
    return tmp;
}

CSVRow const& CSVIterator::operator*() const
{
    return m_row;
}

CSVRow const* CSVIterator::operator->() const
{
    return &m_row;
}

bool CSVIterator::operator==(CSVIterator const& csvIterator)
{
    return ((this == &csvIterator) ||
            ((this->m_str == nullptr) && (csvIterator.m_str == nullptr)));
}

bool CSVIterator::operator!=(CSVIterator const& rhs)
{
    return !((*this) == rhs);
}

CSVRange::CSVRange(std::istream& str)
    : stream(str)
{
}

CSVIterator CSVRange::begin() const
{
    return CSVIterator{ stream };
}

CSVIterator CSVRange::end() const
{
    return CSVIterator{};
}
}
