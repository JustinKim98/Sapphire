// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_COMPUTE_BACKWARD_HPP
#define SAPPHIRE_COMPUTE_COMPUTE_BACKWARD_HPP
#include <Sapphire/tensor/TensorData.hpp>
#include <algorithm>

namespace Sapphire::Compute
{
using namespace TensorUtil;

void Conv2DBackward(TensorData& dx, TensorData& dFilter, const TensorData& dy,
                    const TensorData& x, const TensorData& filter,
                    int strideRow, int strideCol, int dilationRow,
                    int dilationCol, int rowPadding, int columnPadding);

void MaxPool2DBackward(TensorData& dy, TensorData& dx, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding);

void AvgPool2DBackward(TensorData& dy, TensorData& dx, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding);

void DotBackward(TensorData& da, TensorData& db, const TensorData& dy,
                 const TensorData& a, const TensorData& b);

void PowBackward(TensorData& dx, const TensorData& dy, const TensorData& x,
                 float factor);

void cosBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void sinBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void tanBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void coshBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void sinhBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void tanhBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void logBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void log10Backward(TensorData& dx, const TensorData& dy, const TensorData& x);

void ReLUBackward(TensorData& dx, const TensorData& dy);

void LeakyReluBackward(TensorData& dx, const TensorData& dy, float a);

void InverseBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void MeanBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void MeanBackwared(TensorData& dx, const TensorData& dy, const TensorData& x,
                   int dim);

void SoftmaxBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

template <typename Func, typename... Params>
void BroadcastBackwardWith2Inputs(const Shape& shapeOut, const Shape& shapeA,
                                  const Shape& shapeB,
                                  unsigned int totalSizeOut,
                                  unsigned int totalSizeA,
                                  unsigned int totalSizeB,
                                  float* dy, float* da, float* db, float* a,
                                  float* b,
                                  unsigned int shapeIdx,
                                  unsigned int minimumRequiredDim, Func func,
                                  Params ... params)
{
    if (shapeIdx >= shapeOut.Dim() - minimumRequiredDim)
    {
        func(totalSizeOut, da, db, dy, a, b, params...);
        return;
    }

    unsigned int chunkSize = 1;
    while (shapeIdx < shapeOut.Dim() - minimumRequiredDim &&
           (shapeOut.At(shapeIdx) == shapeA.At(shapeIdx) &&
            shapeOut.At(shapeIdx) == shapeB.At(shapeIdx)))
    {
        const auto dim = std::max({ shapeA.At(shapeIdx), shapeB.At(shapeIdx),
                                    shapeOut.At(shapeIdx) });
        chunkSize *= dim;
        shapeIdx += 1;
    }

    //! If given shapes all match together up to minimumRequiredDim, invoke the
    //! kernel directly to improve throughput
    if (shapeIdx >= shapeOut.Dim() - minimumRequiredDim)
    {
        func(totalSizeOut, da, db, dy, a, b, params...);
        return;
    }

    const auto chunkSizeA = chunkSize * shapeA.At(shapeIdx);
    const auto chunkSizeB = chunkSize * shapeB.At(shapeIdx);
    const auto chunkSizeOut = chunkSize * shapeOut.At(shapeIdx);

    const auto strideA = totalSizeA / chunkSizeA;
    const auto strideB = totalSizeB / chunkSizeB;
    const auto strideOut = totalSizeOut / chunkSizeOut;

    const auto maxChunkSize =
        std::max({ chunkSizeOut, chunkSizeA, chunkSizeB });

    for (unsigned int chunkIdx = 0; chunkIdx < maxChunkSize; chunkIdx++)
    {
        BroadcastBackwardWith2Inputs(shapeOut, shapeA, shapeB, strideOut,
                                     strideA,
                                     strideB,
                                     dy + (chunkIdx % chunkSizeOut) *
                                     strideOut,
                                     da + (chunkIdx % chunkSizeA) * strideA,
                                     db + (chunkIdx % chunkSizeB) * strideB,
                                     a + (chunkIdx % chunkSizeA) * strideA,
                                     b + (chunkIdx % chunkSizeB) * strideB,
                                     shapeIdx + 1, minimumRequiredDim, func,
                                     params...);
    }
}
} // namespace Sapphire::Compute

#endif
