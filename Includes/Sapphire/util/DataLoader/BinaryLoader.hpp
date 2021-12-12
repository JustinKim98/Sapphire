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

namespace Sapphire::Util
{
template <typename T>
std::vector<T> GetDataVector(unsigned char* data, std::size_t length)
{
    std::vector<float> dataVector(length / sizeof(T));

    for (int i = 0; i < static_cast<int>(length / sizeof(T)); ++i)
        dataVector[i] = *(reinterpret_cast<T*>(data) + i);

    return dataVector;
}

template <typename T>
class BinaryLoader : public DataLoader<T>
{
public:
    BinaryLoader(std::filesystem::path filePath, std::size_t startOffset,
                 std::size_t numBatches,
                 std::size_t batchStride)
        : DataLoader<T>(std::move(filePath)),
          m_numBatches(numBatches),
          m_startOffset(startOffset),
          m_batchStride(batchStride),
          m_gen(m_rd()),
          m_dist(0, m_numBatches - 1)

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
            free(buffer);
            m_data = vector;
        }
    }

    virtual ~BinaryLoader() = default;

    void LoadData(const Tensor& tensor)
    {
        const auto offset = m_startOffset + m_batchStride * m_dist(m_gen);
        const auto size = tensor.Size();
        std::vector<float> data(size);
        for (int i = 0; i < size; ++i)
            data[i] = m_data[offset + i];
        tensor.LoadData(data);
    }

private:
    std::size_t m_numBatches;
    std::size_t m_startOffset, m_batchStride;
    std::random_device m_rd;
    std::mt19937 m_gen;
    std::uniform_int_distribution<int> m_dist;
    std::vector<T> m_data;
};
}

#endif
