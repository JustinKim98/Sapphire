// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TEST_CONV2D_MODEL_HPP
#define SAPPHIRE_TEST_CONV2D_MODEL_HPP

#include <utility>
#include <vector>

namespace Sapphire::Test
{
void Conv2DModel(std::vector<float> xData, std::vector<float> labelData,
                 int batchSize,
                 int yChannels, int xChannels, std::pair<int, int> xSize,
                 std::pair<int, int> ySize, std::pair<int, int> filterSize,
                 std::pair<int, int> stride, std::pair<int, int> padSize,
                 std::pair<int, int> dilation, float learningRate,
                 bool hostMode, int epochs);
}

#endif
