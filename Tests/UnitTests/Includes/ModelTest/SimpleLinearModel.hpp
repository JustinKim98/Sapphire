// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TEST_SIMPLE_LINEAR_MODEL_HPP
#define SAPPHIRE_TEST_SIMPLE_LINEAR_MODEL_HPP
#include <vector>

namespace Sapphire::Test
{
void SimpleLinearModel(std::vector<float> xData, std::vector<float> labelData, int inputSize,
                       int outputSize, float learningRate, int batchSize, int epochs);
}
#endif
