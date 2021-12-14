// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_CSV_LOADER_HPP
#define SAPPHIRE_UTIL_CSV_LOADER_HPP

#include <Sapphire/util/DataLoader/DataLoader.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <iterator>
#include <vector>
#include <deque>
#include <string>
#include <random>
#include <charconv>
#include <fstream>
#include <functional>

namespace Sapphire::Util
{
class CSVRow
{
public:
    std::string_view operator[](std::size_t index) const;

    [[nodiscard]] std::size_t size() const;

    void ReadNextRow(std::istream& str);

private:
    std::string m_line;
    std::vector<int> m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data);

class CSVIterator
{
public:
    typedef std::input_iterator_tag iterator_category;
    typedef CSVRow value_type;
    typedef std::size_t difference_type;
    typedef CSVRow* pointer;
    typedef CSVRow& reference;

    CSVIterator(std::istream& str);

    CSVIterator() = default;

    CSVIterator& operator++();

    CSVIterator operator++(int);

    CSVRow const& operator*() const;

    CSVRow const* operator->() const;

    bool operator==(CSVIterator const& csvIterator);

    bool operator!=(CSVIterator const& rhs);

private:
    std::istream* m_str = nullptr;
    CSVRow m_row;
};

class CSVRange
{
public:
    CSVRange(std::istream& str);

    [[nodiscard]] CSVIterator begin() const;

    [[nodiscard]] CSVIterator end() const;

private:
    std::istream& stream;
};

template <typename T>
class CsvLoader : public DataLoader<T>
{
public:
    CsvLoader(std::filesystem::path filePath)
        : DataLoader<T>(std::move(filePath)),
          m_gen(m_rd())
    {
        std::ifstream file;
        file.open(DataLoader<T>::m_filePath);
        if (file.is_open())
        {
            for (auto& row : CSVRange(file))
            {
                m_csvRows.emplace_back(row);
            }
            file.close();
        }
        else
            throw std::runtime_error("CsvLoader - Could not open file (" +
                                     filePath.string() + ")");
    }

    void LoadData(const Tensor& tensor, std::size_t lineIdx,
                  std::size_t firstElemIdx, std::size_t lastElemIdx)
    {
        if (lineIdx >= m_csvRows.size())
            throw std::invalid_argument(
                "CsvLoader::LoadData - line index (" + std::to_string(lineIdx) +
                ") exceeds number of rows in the file (" +
                std::to_string(m_csvRows.size()) + ")");
        const auto csvRow = m_csvRows.at(lineIdx);
        const auto size = tensor.Size();
        std::vector<float> data(size);
        for (std::size_t i = 0; i <= lastElemIdx - firstElemIdx; ++i)
        {
            T value = static_cast<T>(0.0f);
            const auto elemIdx = i + firstElemIdx;
            std::string_view str = csvRow[elemIdx];
            std::from_chars(str.data(), str.data() + str.size(),
                            value);
            data[i] = static_cast<float>(value);
        }
        tensor.LoadData(data);
    }

    void LoadData(const Tensor& tensor, std::vector<std::size_t> lineIndices,
                  std::size_t firstIdx, std::size_t lastIdx,
                  std::function<std::vector<float>(std::vector<T>)> preprocess)
    {
        const auto inputSizePerBatch = lastIdx - firstIdx + 1;
        const auto batchSize = lineIndices.size();
        std::vector<T> data(batchSize * inputSizePerBatch);

        for (std::size_t batchIdx = 0;
             batchIdx < lineIndices.size(); ++batchIdx)
        {
            const auto csvRow = m_csvRows.at(lineIndices.at(batchIdx));
            for (std::size_t i = 0; i < inputSizePerBatch; ++i)
            {
                T value = static_cast<T>(0.0f);
                const auto elemIdx = i + firstIdx;
                std::string_view str = csvRow[elemIdx];
                std::from_chars(str.data(), str.data() + str.size(), value);
                data[batchIdx * inputSizePerBatch + i] = value;
            }
        }
        tensor.LoadData(preprocess(data));
    }

    [[nodiscard]] std::size_t GetLineSize() const
    {
        return m_csvRows.size();
    }

private:
    std::random_device m_rd;
    std::mt19937 m_gen;
    std::deque<CSVRow> m_csvRows;
};
}

#endif
