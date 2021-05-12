// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_COMPUTE_DECL_HPP
#define Sapphire_COMPUTE_COMPUTE_DECL_HPP

#include <Sapphire/tensor/TensorData.hpp>
#include <algorithm>
#include <vector>

namespace Sapphire::Compute
{
using namespace TensorUtil;

//! Performs out = a + b
void Add(TensorData& out, const TensorData& a, const TensorData& b);

//! Performs out = a - b
void Sub(TensorData& out, const TensorData& a, const TensorData& b);

//! Performs GEMM (out = a*b + c)
void Gemm(TensorData& out, const TensorData& a, const TensorData& b,
          const TensorData& c);

//! Performs GEMM (out = a*b + c) using the sparse matrix
void SparseGemm(TensorData& out, const TensorData& a, const TensorData& b,
                TensorData& c);

//! Performs output = input*factor
void Scale(TensorData& output, const TensorData& input, float factor);

//! Performs output = TransposeKernel(input)
void Transpose(TensorData& output, const TensorData& input);

//! Performs Element-wise multiply
void Dot(TensorData& out, const TensorData& a, const TensorData& b);

//! Performs out = input^factor for each element
void Pow(TensorData& out, const TensorData& input, float factor);

void cos(TensorData& out, const TensorData& input);

void sin(TensorData& out, const TensorData& input);

void tan(TensorData& out, const TensorData& input);

void cosh(TensorData& out, const TensorData& input);

void sinh(TensorData& out, const TensorData& input);

void tanh(TensorData& out, const TensorData& input);

void log(TensorData& out, const TensorData& input);

void log10(TensorData& out, const TensorData& input);

void ReLU(TensorData& out, const TensorData& input);

void ReLUDerivative(TensorData& out, const TensorData& input);

void LeakyReLU(TensorData& out, const TensorData& input, float a);

void LeakyReluDerivative(TensorData& out, const TensorData& input, float a);

void Inverse(TensorData& out, const TensorData& input);

void Mean(TensorData& out, const TensorData& x);

void Mean(TensorData& out, const TensorData& input, int dim);

void Softmax(TensorData& out, const TensorData& x);

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
                          Params... params)
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
        auto dim = std::max({ shapeA.At(shapeIdx), shapeB.At(shapeIdx),
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
                          float* out, float* A, float* B, unsigned int shapeIdx,
                          unsigned int minimumRequiredDim, Func func,
                          Params... params)
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
        auto dim = std::max({ shapeA.At(shapeIdx), shapeB.At(shapeIdx),
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

}  // namespace Sapphire::Compute

#endif
