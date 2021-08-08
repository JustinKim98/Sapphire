// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_BROADCAST_HPP
#define SAPPHIRE_COMPUTE_BROADCAST_HPP
#include <Sapphire/tensor/Shape.hpp>
#include <algorithm>

namespace Sapphire::Compute
{
//! Broadcasts given shape and invokes the function
//! Each shape variable are required to be same size in reversed order
//! containing row and column indices shapes must be padded to match the same
//! shape as yShape
//! shapeIdx starts at last index of the shape array
//! totalSize parameters should contain actual total size of the whole array
//! including batch size
template <typename Func, typename... Params>
void BroadcastWith3Inputs(const Shape& yShape, const Shape& aShape,
                          const Shape& bShape, const Shape& cShape,
                          unsigned int totalSizeOut, unsigned int totalSizeA,
                          unsigned int totalSizeB, unsigned int totalSizeC,
                          float* out, const float* A, const float* B,
                          const float* C,
                          unsigned int shapeIdx,
                          unsigned int minimumRequiredDim, Func func,
                          Params ... params)
{
    if (shapeIdx >= yShape.Dim() - minimumRequiredDim)
    {
        func(totalSizeOut, out, A, B, C, params...);
        return;
    }

    unsigned int chunkSize = 1;
    while (shapeIdx < yShape.Dim() - minimumRequiredDim &&
           yShape.At(shapeIdx) == aShape.At(shapeIdx) &&
           yShape.At(shapeIdx) == bShape.At(shapeIdx) &&
           yShape.At(shapeIdx) == cShape.At(shapeIdx))
    {
        const auto dim = yShape.At(shapeIdx);
        chunkSize *= dim;
        shapeIdx += 1;
    }

    //! ChunkSize represents the total size of the leftover parts to be calculated for each tensor
    const auto chunkSizeA = chunkSize * aShape.At(shapeIdx);
    const auto chunkSizeB = chunkSize * bShape.At(shapeIdx);
    const auto chunkSizeC = chunkSize * cShape.At(shapeIdx);
    const auto chunkSizeOut = chunkSize * yShape.At(shapeIdx);

    const auto strideA = totalSizeA / chunkSizeA;
    const auto strideB = totalSizeB / chunkSizeB;
    const auto strideC = totalSizeC / chunkSizeC;
    const auto strideOut = totalSizeOut / chunkSizeOut;

    const auto maxChunkSize =
        std::max({ chunkSizeOut, chunkSizeA, chunkSizeB, chunkSizeC });

    //! Broadcasting happens here by incrementing tensors with dimension 1 with smaller chinkSize
    for (unsigned int chunkIdx = 0; chunkIdx < maxChunkSize; chunkIdx++)
    {
        BroadcastWith3Inputs(yShape, aShape, bShape, cShape, strideOut,
                             strideA, strideB, strideC,
                             out + (chunkIdx % chunkSizeOut) * strideOut,
                             A + (chunkIdx % chunkSizeA) * strideA,
                             B + (chunkIdx % chunkSizeB) * strideB,
                             C + (chunkIdx % chunkSizeC) * strideC,
                             shapeIdx + 1, minimumRequiredDim, func, params...);
    }
}

template <typename Func, typename... Params>
void BroadcastWith2Inputs(const Shape& yShape, const Shape& aShape,
                          const Shape& bShape, unsigned int totalSizeOut,
                          unsigned int totalSizeA, unsigned int totalSizeB,
                          float* out, const float* A, const float* B,
                          unsigned int shapeIdx,
                          unsigned int minimumRequiredDim, Func func,
                          Params ... params)
{
    if (shapeIdx >= yShape.Dim() - minimumRequiredDim)
    {
        func(totalSizeOut, out, A, B, params...);
        return;
    }

    unsigned int chunkSize = 1;
    while (shapeIdx < yShape.Dim() - minimumRequiredDim &&
           (yShape.At(shapeIdx) == aShape.At(shapeIdx) &&
            yShape.At(shapeIdx) == bShape.At(shapeIdx)))
    {
        const auto dim = yShape.At(shapeIdx);
        chunkSize *= dim;
        shapeIdx += 1;
    }

    const auto chunkSizeA = chunkSize * aShape.At(shapeIdx);
    const auto chunkSizeB = chunkSize * bShape.At(shapeIdx);
    const auto chunkSizeOut = chunkSize * yShape.At(shapeIdx);

    const auto strideA = totalSizeA / chunkSizeA;
    const auto strideB = totalSizeB / chunkSizeB;
    const auto strideOut = totalSizeOut / chunkSizeOut;

    const auto maxChunkSize =
        std::max({ chunkSizeOut, chunkSizeA, chunkSizeB });

    for (unsigned int chunkIdx = 0; chunkIdx < maxChunkSize; chunkIdx++)
    {
        BroadcastWith2Inputs(yShape, aShape, bShape, strideOut, strideA,
                             strideB,
                             out + (chunkIdx % chunkSizeOut) * strideOut,
                             A + (chunkIdx % chunkSizeA) * strideA,
                             B + (chunkIdx % chunkSizeB) * strideB,
                             shapeIdx + 1, minimumRequiredDim, func, params...);
    }
}

template <typename Func, typename... Params>
void BroadcastBackwardWith2Inputs(
    const Shape& yShape, const Shape& aShape, const Shape& bShape,
    unsigned int totalSizeOut, unsigned int totalSizeA, unsigned int totalSizeB,
    const float* dy, float* da, float* db, const float* a, const float* b,
    unsigned int shapeIdx,
    unsigned int minimumRequiredDim, Func func, Params ... params)
{
    if (shapeIdx >= yShape.Dim() - minimumRequiredDim)
    {
        func(totalSizeOut, da, db, dy, a, b, params...);
        return;
    }

    unsigned int chunkSize = 1;
    while (shapeIdx < yShape.Dim() - minimumRequiredDim &&
           (yShape.At(shapeIdx) == aShape.At(shapeIdx) &&
            yShape.At(shapeIdx) == bShape.At(shapeIdx)))
    {
        const auto dim = yShape.At(shapeIdx);
        chunkSize *= dim;
        shapeIdx += 1;
    }

    const auto chunkSizeA = chunkSize * aShape.At(shapeIdx);
    const auto chunkSizeB = chunkSize * bShape.At(shapeIdx);
    const auto chunkSizeOut = chunkSize * yShape.At(shapeIdx);

    const auto strideA = totalSizeA / chunkSizeA;
    const auto strideB = totalSizeB / chunkSizeB;
    const auto strideOut = totalSizeOut / chunkSizeOut;

    const auto maxChunkSize =
        std::max({ chunkSizeOut, chunkSizeA, chunkSizeB });

    for (unsigned int chunkIdx = 0; chunkIdx < maxChunkSize; chunkIdx++)
    {
        BroadcastBackwardWith2Inputs(
            yShape, aShape, bShape, strideOut, strideA, strideB,
            dy + (chunkIdx % chunkSizeOut) * strideOut,
            da + (chunkIdx % chunkSizeA) * strideA,
            db + (chunkIdx % chunkSizeB) * strideB,
            a + (chunkIdx % chunkSizeA) * strideA,
            b + (chunkIdx % chunkSizeB) * strideB, shapeIdx + 1,
            minimumRequiredDim, func, params...);
    }
}
}

#endif
