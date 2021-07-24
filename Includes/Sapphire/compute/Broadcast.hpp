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
//! shape as shapeOut
//! shapeIdx starts at last index of the shape array
//! totalSize parameters should contain actual total size of the whole array
//! including batch size
template <typename Func, typename... Params>
void BroadcastWith3Inputs(const Shape& shapeOut, const Shape& shapeA,
                          const Shape& shapeB, const Shape& shapeC,
                          unsigned int totalSizeOut, unsigned int totalSizeA,
                          unsigned int totalSizeB, unsigned int totalSizeC,
                          float* out, const float* A, const float* B,
                          const float* C,
                          unsigned int shapeIdx,
                          unsigned int minimumRequiredDim, Func func,
                          Params ... params)
{
    if (shapeIdx >= shapeOut.Dim() - minimumRequiredDim)
    {
        func(totalSizeOut, out, A, B, C, params...);
        return;
    }

    unsigned int chunkSize = 1;
    while (shapeIdx < shapeOut.Dim() - minimumRequiredDim &&
           shapeOut.At(shapeIdx) == shapeA.At(shapeIdx) &&
           shapeOut.At(shapeIdx) == shapeB.At(shapeIdx) &&
           shapeOut.At(shapeIdx) == shapeC.At(shapeIdx))
    {
        const auto dim =
            std::max({ shapeA.At(shapeIdx), shapeB.At(shapeIdx),
                       shapeC.At(shapeIdx), shapeOut.At(shapeIdx) });
        chunkSize *= dim;
        shapeIdx += 1;
    }

    //! If given shapes all match together up to minimumRequiredDim, invoke the
    //! kernel directly to improve throughput
    if (shapeIdx >= shapeOut.Dim() - minimumRequiredDim)
    {
        func(totalSizeOut, out, A, B, C, params...);
        return;
    }

    const auto chunkSizeA = chunkSize * shapeA.At(shapeIdx);
    const auto chunkSizeB = chunkSize * shapeB.At(shapeIdx);
    const auto chunkSizeC = chunkSize * shapeC.At(shapeIdx);
    const auto chunkSizeOut = chunkSize * shapeOut.At(shapeIdx);

    const auto strideA = totalSizeA / chunkSizeA;
    const auto strideB = totalSizeB / chunkSizeB;
    const auto strideC = totalSizeC / chunkSizeC;
    const auto strideOut = totalSizeOut / chunkSizeOut;

    const auto maxChunkSize =
        std::max({ chunkSizeOut, chunkSizeA, chunkSizeB, chunkSizeC });

    for (unsigned int chunkIdx = 0; chunkIdx < maxChunkSize; chunkIdx++)
    {
        BroadcastWith3Inputs(shapeOut, shapeA, shapeB, shapeC, strideOut,
                             strideA, strideB, strideC,
                             out + (chunkIdx % chunkSizeOut) * strideOut,
                             A + (chunkIdx % chunkSizeA) * strideA,
                             B + (chunkIdx % chunkSizeB) * strideB,
                             C + (chunkIdx % chunkSizeC) * strideC,
                             shapeIdx + 1, minimumRequiredDim, func, params...);
    }
}

template <typename Func, typename... Params>
void BroadcastWith2Inputs(const Shape& shapeOut, const Shape& shapeA,
                          const Shape& shapeB, unsigned int totalSizeOut,
                          unsigned int totalSizeA, unsigned int totalSizeB,
                          float* out, const float* A, const float* B,
                          unsigned int shapeIdx,
                          unsigned int minimumRequiredDim, Func func,
                          Params ... params)
{
    if (shapeIdx >= shapeOut.Dim() - minimumRequiredDim)
    {
        func(totalSizeOut, out, A, B, params...);
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
        func(totalSizeOut, out, A, B, params...);
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
        BroadcastWith2Inputs(shapeOut, shapeA, shapeB, strideOut, strideA,
                             strideB,
                             out + (chunkIdx % chunkSizeOut) * strideOut,
                             A + (chunkIdx % chunkSizeA) * strideA,
                             B + (chunkIdx % chunkSizeB) * strideB,
                             shapeIdx + 1, minimumRequiredDim, func, params...);
    }
}

template <typename Func, typename... Params>
void BroadcastBackwardWith2Inputs(
    const Shape& shapeOut, const Shape& shapeA, const Shape& shapeB,
    unsigned int totalSizeOut, unsigned int totalSizeA, unsigned int totalSizeB,
    const float* dy, float* da,float* db, const float* a, const float* b, unsigned int shapeIdx,
    unsigned int minimumRequiredDim, Func func, Params ... params)
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
        BroadcastBackwardWith2Inputs(
            shapeOut, shapeA, shapeB, strideOut, strideA, strideB,
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
