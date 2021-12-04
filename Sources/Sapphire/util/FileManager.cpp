// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/util/FileManager.hpp>
#include <fstream>
#include <vector>

namespace Sapphire
{

void WriteToFile(std::string filePath, unsigned char* data,
                 std::size_t data_len)
{
    std::ofstream fout;
    fout.open(filePath, std::ios::out | std::ios::binary);

    if (fout.is_open())
    {
        fout.write((const char*)data, data_len);
        fout.close();
    }
    return;
}
}
