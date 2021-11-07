// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/naive/Pool.hpp>
#include <limits>

namespace Sapphire::Compute::Dense::Naive
{
void MaxPool2D(TensorUtil::TensorData& y, const TensorUtil::TensorData& x,
               std::pair<int, int> filterSize,
               std::pair<int, int> stride,
               std::pair<int, int> padding, std::pair<int, int> dilation)
{
    const auto [filterRows, filterCols] = filterSize;
    const auto [rowStride, colStride] = stride;
    const auto [rowPadding, colPadding] = padding;
    const auto [rowDilation, colDilation] = dilation;
    const auto xUnitSize = x.GetUnitSize(3);
    const auto yUnitSize = y.GetUnitSize(3);
    const auto batchSize = x.GetNumUnits(3);
    const auto xShape = x.GetShape();
    const auto yShape = y.GetShape();

    for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (int channelIdx = 0; channelIdx < xShape.At(-3); ++channelIdx)
        {
            const auto xOffset =
                batchIdx * xUnitSize + channelIdx * x.GetUnitSize(2);
            const auto yOffset =
                batchIdx * yUnitSize + channelIdx * y.GetUnitSize(2);

            for (int yRowIdx = 0; yRowIdx < yShape.At(-2); ++yRowIdx)
                for (int yColIdx = 0; yColIdx < yShape.At(-1); ++yColIdx)
                {
                    const auto xRowOffset = yRowIdx * rowStride - rowPadding;
                    const auto xColOffset = yColIdx * colStride - colPadding;
                    float maxVal = -std::numeric_limits<float>::max();

                    for (int filterRowIdx = 0; filterRowIdx < filterRows;
                         ++filterRowIdx)
                        for (int filterColIdx = 0; filterColIdx < filterCols;
                             ++filterColIdx)
                        {
                            const auto xRowIdx =
                                xRowOffset + filterRowIdx * rowDilation;
                            const auto xColIdx =
                                xColOffset + filterColIdx * colDilation;
                            if (xRowIdx < 0.0f || xColIdx < 0.0 || xRowIdx >=
                                xShape.At(-2) || xColIdx >= xShape.At(-1))
                                continue;

                            const auto val =
                                x.HostRawPtr()
                                [xOffset + xRowIdx * xShape.At(-1) + xColIdx];
                            if (val > maxVal)
                                maxVal = val;
                        }

                    y.HostMutableRawPtr()[yOffset + yRowIdx * yShape.At(-1) +
                                          yColIdx] = maxVal;
                }
        }
}

void MaxPool2DBackward(TensorUtil::TensorData& dx,
                       const TensorUtil::TensorData& x,
                       const TensorUtil::TensorData& dy,
                       std::pair<int, int> filterSize,
                       std::pair<int, int> stride, std::pair<int, int> padding,
                       std::pair<int, int> dilation)
{
    const auto [filterRows, filterCols] = filterSize;
    const auto [rowStride, colStride] = stride;
    const auto [rowPadding, colPadding] = padding;
    const auto [rowDilation, colDilation] = dilation;
    const auto dxUnitSize = dx.GetUnitSize(3);
    const auto dyUnitSize = dy.GetUnitSize(3);
    const auto batchSize = dx.GetNumUnits(3);
    const auto dxShape = dx.GetShape();
    const auto dyShape = dy.GetShape();

    for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        for (int channelIdx = 0; channelIdx < dxShape.At(-3); ++channelIdx)
        {
            const auto dxOffset =
                batchIdx * dxUnitSize + channelIdx * dx.GetUnitSize(2);
            const auto dyOffset =
                batchIdx * dyUnitSize + channelIdx * dy.GetUnitSize(2);

            for (int yRowIdx = 0; yRowIdx < dyShape.At(-2); ++yRowIdx)
                for (int yColIdx = 0; yColIdx < dyShape.At(-1); ++yColIdx)
                {
                    const auto xRowOffset = yRowIdx * rowStride - rowPadding;
                    const auto xColOffset = yColIdx * colStride - colPadding;

                    float maxVal = -std::numeric_limits<float>::max();
                    int maxXRowIdx = -1;
                    int maxXColIdx = -1;

                    for (int filterRowIdx = 0; filterRowIdx < filterRows;
                         ++filterRowIdx)
                        for (int filterColIdx = 0; filterColIdx < filterCols;
                             ++filterColIdx)
                        {
                            const auto xRowIdx =
                                xRowOffset + filterRowIdx * rowDilation;
                            const auto xColIdx =
                                xColOffset + filterColIdx * colDilation;

                            if (xRowIdx < 0.0f || xColIdx < 0.0 ||
                                xRowIdx >= dxShape.At(-2) ||
                                xColIdx >= dxShape.At(-1))
                                continue;

                            const auto val =
                                x.HostRawPtr()[dxOffset +
                                               xRowIdx * dxShape.At(-1) +
                                               xColIdx];
                            if (val > maxVal)
                            {
                                maxVal = val;
                                maxXRowIdx = xRowIdx;
                                maxXColIdx = xColIdx;
                            }
                        }

                    dx.HostMutableRawPtr()[dxOffset +
                                           maxXRowIdx * dxShape.At(-1) +
                                           maxXColIdx] +=
                        dy.HostRawPtr()[dyOffset + yRowIdx * dyShape.At(-1) +
                                        yColIdx];
                }
        }
}
}