// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_COMPUTE_DECL_HPP
#define MOTUTAPU_COMPUTE_COMPUTE_DECL_HPP

#include <Motutapu/tensor/TensorData.hpp>
#include <vector>

namespace Motutapu::Compute
{
//! Performs out = out + add

using namespace TensorUtil;

//! Performs out = a + b
void Add(TensorData& out, const TensorData& a, const TensorData& b);

//! Performs out = out - sub
void Sub(TensorData& out, const TensorData& sub);

//! Performs out = a - b

void Sub(TensorData& out, const TensorData& a, const TensorData& b);

//! Performs GEMM (out = a*b + c)
void Gemm(TensorData& out, const TensorData& a, const TensorData& b,
          const TensorData& c);

//! Performs output = input*factor
void Scale(TensorData& output, const TensorData& input, float factor);

//! Performs output = TransposeKernel(input)
void Transpose(TensorData& output, const TensorData& input);

//! Performs out = out^factor for each element
void Pow(TensorData& out, int factor);

//! Performs out = a^factor for each element
void Pow(TensorData& out, const TensorData& a, int factor);

//! Broadcasts given shape and invokes the function
//! Each shape variable are required to be same size in reversed order
//! containing row and column indices shapes must be padded to match the same
//! shape as shapeOut
//! shapeIdx starts at last index of the shape array
//! totalSize parameters should contain actual total size of the whole array
//! including batch size
template <typename Func, typename... Params>
void broadcastWith3Inputs(Shape shapeOut, Shape shapeA, Shape shapeB,
                          Shape shapeC, unsigned int totalSizeOut,
                          unsigned int totalSizeA, unsigned int totalSizeB,
                          unsigned int totalSizeC, float* out, float* A,
                          float* B, float* C, unsigned int shapeIdx,
                          unsigned int minimumRequiredDim, Func func,
                          Params... params)
{
    if (shapeIdx == 0)
    {
        shapeA.Expand(shapeOut.Dim());
        shapeB.Expand(shapeOut.Dim());
        shapeC.Expand(shapeOut.Dim());
    }

    if (shapeIdx >= shapeOut.Dim() - minimumRequiredDim)
    {
        func(totalSizeOut, out, A, B, C, params...);
        return;
    }

    unsigned int batchSize = 1;
    while (!((shapeA[shapeIdx] == 1 || shapeB[shapeIdx] == 1 ||
              shapeC[shapeIdx] == 1) &&
             shapeOut[shapeIdx] != 1) &&
           shapeIdx < shapeOut.Dim() - minimumRequiredDim)
    {
        batchSize *= shapeOut[shapeIdx];
        shapeIdx += 1;
    }

    const auto chunkSizeA = batchSize * shapeA[shapeIdx];
    const auto chunkSizeB = batchSize * shapeB[shapeIdx];
    const auto chunkSizeC = batchSize * shapeC[shapeIdx];
    const auto chunkSizeOut = batchSize * shapeOut[shapeIdx];

    const auto strideA = totalSizeA / chunkSizeA;
    const auto strideB = totalSizeB / chunkSizeB;
    const auto strideC = totalSizeC / chunkSizeC;
    const auto strideOut = totalSizeOut / chunkSizeOut;

    for (unsigned int chunkIdx = 0; chunkIdx < chunkSizeOut; chunkIdx++)
    {
        broadcastWith3Inputs(shapeOut, shapeA, shapeB, shapeC, strideOut,
                             strideA, strideB, strideC,
                             out + chunkIdx * strideOut,
                             A + (chunkIdx % chunkSizeA) * strideA,
                             B + (chunkIdx % chunkSizeB) * strideB,
                             C + (chunkIdx % chunkSizeC) * strideC,
                             shapeIdx + 1, minimumRequiredDim, func, params...);
    }
}
}  // namespace Motutapu::Compute

#endif
