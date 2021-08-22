// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/IndexingOps.hpp>

namespace Sapphire::Compute
{
void Reshape(TensorUtil::TensorData& tensorData, Shape newShape)
{
    const Shape tensorShape = tensorData.GetShape();

    const auto padUnitSize = static_cast<unsigned long>(32 / sizeof(float));

    const auto newPaddedColSize =
        newShape.Cols() % padUnitSize == 0
            ? newShape.Cols()
            : newShape.Cols() / padUnitSize * padUnitSize + padUnitSize;

    auto* tempData = new float[tensorShape.Size()];
    for (std::size_t ii = 0; ii < tensorData.GetBatchSize(1); ++ii)
        for (std::size_t i = 0; i < tensorData.Cols(); ++i)
        {
            const auto data =
                tensorData
                    .GetDenseHost()[ii * tensorData.PaddedHostColSize + i];
            tempData[ii * tensorShape.Cols() + i] = data;
        }

    tensorData.PaddedHostColSize = newPaddedColSize;
    tensorData.TensorShape = newShape;

    for (std::size_t ii = 0; ii < tensorData.GetBatchSize(1); ++ii)
        for (std::size_t i = 0; i < tensorData.Cols(); ++i)
        {
            const auto data = tempData[ii * newShape.Cols() + i];
            tensorData
                .GetMutableDenseHost()[ii * tensorData.PaddedHostColSize + i] =
                data;
        }

    delete[] tempData;
}

}