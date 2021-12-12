// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_UTIL_DATA_LOADER_HPP
#define SAPPHIRE_UTIL_DATA_LOADER_HPP

#include <filesystem>

namespace Sapphire::Util
{
template <typename T>
class DataLoader
{
public:
    DataLoader(std::filesystem::path filePath)
        : m_filePath(filePath)
    {
    }

protected:
    std::filesystem::path m_filePath;
};
}

#endif
