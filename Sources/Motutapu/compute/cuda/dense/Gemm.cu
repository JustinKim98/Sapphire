// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any size_tellectual
// property of any third parties.

#include <Motutapu/compute/cuda/dense/DenseMatmul.hpp>

namespace Motutapu::Cuda::Dense
{
//! M, N, K must be smaller than 64
__global__ void GemmHalf(half* out, half* A, half* chunkPtrB, size_t M,
                         size_t N, size_t K,
                         size_t paddedColSizeA, size_t paddedColSizeB,
                         size_t paddedColSizeOut, size_t size)
{
    __shared__ half matrixA[64][64 + 1];
    __shared__ half matrixB[64][64 + 1];

    size_t rowIdx = threadIdx.x;
    size_t colIdx = threadIdx.y;

    
}

}
