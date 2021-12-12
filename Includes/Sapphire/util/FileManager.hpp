// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_FILEMANAGER_HPP
#define SAPPHIRE_FILEMANAGER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

namespace Sapphire
{
template <typename T>
std::vector<T> ReadFile(std::string filePath)
{
    std::ifstream is(filePath, std::ifstream::binary);
    if (is)
    {
        is.seekg(0, is.end);
        int length = (int)is.tellg();
        is.seekg(0, is.beg);

        auto* buffer = (unsigned char*)malloc(length);

        is.read((char*)buffer, length);
        is.close();

        std::vector<T> dataVector(length / sizeof(T));

        for (int i = 0; i < static_cast<int>(length / sizeof(T)); ++i)
            dataVector[i] = *(reinterpret_cast<T*>(buffer) + i);

        return dataVector;
    }
    throw std::runtime_error("ReadFile - Failed to open file : " + filePath);
}

void WriteToFile(std::string filePath, unsigned char* data,
                 std::size_t data_len);

} // namespace Sapphire::NN

#endif
