// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TEST_SIMPLE_LINEAR_MODEL_HPP
#define SAPPHIRE_TEST_SIMPLE_LINEAR_MODEL_HPP

#include <filesystem>

namespace Sapphire::Test
{
void MnistLinear(std::filesystem::path filePath, int batchSize,
                       float learningRate, int epochs, bool hostMode);
}
#endif
