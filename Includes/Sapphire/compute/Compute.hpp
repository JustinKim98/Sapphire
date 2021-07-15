// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_COMPUTE_DECL_HPP
#define SAPPHIRE_COMPUTE_COMPUTE_DECL_HPP

#include <Sapphire/tensor/TensorData.hpp>
#include <algorithm>
#include <vector>

namespace Sapphire::Compute
{
using namespace TensorUtil;

//! Warning
//! These operations does not check validity of the inputs
//! If input data condition does not meet, it will cause unhandled errors

//! Performs y = a + b
void Add(TensorData& y, const TensorData& a, const TensorData& b);

//! Performs y = a - b
void Sub(TensorData& y, const TensorData& a, const TensorData& b);

//! Performs GEMM (y = a*b + c)
void Gemm(TensorData& y, const TensorData& a, const TensorData& b,
          const TensorData& c);

//! Performs GEMM (y = a*b + c) using the sparse matrix
void SparseGemm(TensorData& y, const TensorData& a, const TensorData& b,
                TensorData& c);

void Conv2DForward(TensorData& y, const TensorData& x,
                   const TensorData& Filter, int strideRow, int strideCol,
                   int dilationRow, int dilationCol, int rowPadding,
                   int columnPadding);

void Conv2DBackward(TensorData& dx, TensorData& dFilter, const TensorData& dy,
                    const TensorData& x, const TensorData& filter,
                    int strideRow, int strideCol, int dilationRow,
                    int dilationCol, int rowPadding, int columnPadding);

void MaxPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding);

void MaxPool2DBackward(TensorData& dy, TensorData& dx, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding);

void AvgPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding);

void AvgPool2DBackward(TensorData& dy, TensorData& dx, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding);

//! Performs y = x*factor
void Scale(TensorData& y, const TensorData& x, float factor);

//! Performs y = TransposeKernel(x)
void Transpose(TensorData& y, const TensorData& x);

//! Performs Element-wise multiply
void Dot(TensorData& y, const TensorData& a, const TensorData& b);

//! Performs y = x^factor for each element
void Pow(TensorData& y, const TensorData& x, float factor);

void cos(TensorData& y, const TensorData& x);

void sin(TensorData& y, const TensorData& x);

void tan(TensorData& y, const TensorData& x);

void cosh(TensorData& y, const TensorData& x);

void sinh(TensorData& y, const TensorData& x);

void tanh(TensorData& y, const TensorData& x);

void log(TensorData& y, const TensorData& x);

void log10(TensorData& y, const TensorData& x);

void ReLU(TensorData& y, const TensorData& x);

void ReLUBackward(TensorData& dx, const TensorData& dy);

void LeakyReLU(TensorData& y, const TensorData& x, float a);

void LeakyReluBackward(TensorData& dx, const TensorData& dy, float a);

void Inverse(TensorData& y, const TensorData& x);

void InverseBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void Mean(TensorData& y, const TensorData& x);

void Mean(TensorData& y, const TensorData& x, int dim);

void Softmax(TensorData& y, const TensorData& x);

void SoftmaxBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

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
                          float* out, float* A, float* B, float* C,
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
        const auto dim = std::max({ shapeA.At(shapeIdx), shapeB.At(shapeIdx),
                                    shapeC.At(shapeIdx),
                                    shapeOut.At(shapeIdx) });
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
                          float* out, float* A, float* B, unsigned int shapeIdx,
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
} // namespace Sapphire::Compute

#endif
