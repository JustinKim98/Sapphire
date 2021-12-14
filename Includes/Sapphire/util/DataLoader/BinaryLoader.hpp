// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_BINARY_LOADER_HPP
#define SAPPHIRE_UTIL_BINARY_LOADER_HPP

#include <Sapphire/util/DataLoader/DataLoader.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <random>
#include <fstream>
#include <functional>

namespace Sapphire::Util
{
template <typename T>
std::vector<T> GetDataVector(char* data, std::size_t fileByteSize)
{
    std::vector<T> dataVector(fileByteSize / sizeof(T));

    for (int i = 0; i < static_cast<int>(fileByteSize / sizeof(T)); ++i)
        dataVector[i] = *(reinterpret_cast<T*>(data) + i);

    return dataVector;
}

template <typename T>
class BinaryLoader : public DataLoader<T>
{
public:
    BinaryLoader(std::filesystem::path filePath, std::size_t startOffset,
                 std::size_t totalByteSize,
                 std::size_t batchSize,
                 std::size_t batchStride)
        : DataLoader<T>(std::move(filePath)),
          m_batchSize(batchSize),
          m_totalByteSize(totalByteSize),
          m_startOffset(startOffset),
          m_batchStride(batchStride),
          m_gen(m_rd()),
          m_dist(0, m_batchSize - 1)

    {
        std::ifstream is(DataLoader<T>::m_filePath, std::ifstream::binary);
        if (is)
        {
            is.seekg(0, is.end);
            std::size_t size = is.tellg();
            is.seekg(0, is.beg);
            auto* buffer = static_cast<char*>(malloc(size));
            is.read(buffer, size);
            is.close();

            auto vector = GetDataVector<T>(buffer, size);
            m_fileData = std::move(vector);
            free(buffer);
        }
    }

    virtual ~BinaryLoader() = default;

    void LoadData(const Tensor& tensor, std::vector<std::size_t> batchIndices,
                  std::size_t firstIdx, std::size_t lastIdx,
                  std::function<std::vector<float>(std::vector<T>)> preprocess)
    {
        const auto inputSizePerBatch = lastIdx - firstIdx + 1;
        const auto batchSize = batchIndices.size();

        std::vector<T> data(batchSize * inputSizePerBatch);

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (std::size_t i = 0; i < inputSizePerBatch; ++i)
            {
                const auto fileIdx =
                    m_startOffset + batchIndices.at(batchIdx) * m_batchStride +
                    firstIdx + i;
                data.at(batchIdx * inputSizePerBatch + i) = m_fileData.at(
                    fileIdx);
            }
        }
        tensor.LoadData(preprocess(data));
    }

private:
    std::size_t m_batchSize, m_totalByteSize;
    std::size_t m_startOffset, m_batchStride;
    std::random_device m_rd;
    std::mt19937 m_gen;
    std::uniform_int_distribution<int> m_dist;
    std::vector<T> m_fileData;
};
}

#endif
