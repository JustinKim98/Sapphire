// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TEST_CONV2D_MODEL_HPP
#define SAPPHIRE_TEST_CONV2D_MODEL_HPP

#include <filesystem>

namespace Sapphire::Test
{
void Conv2DModelTest(
    std::filesystem::path filePath,
    int batchSize,
    float learningRate, bool hostMode, int epochs);
}

#endif
