// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any size_tellectual
// property of any third parties.

#ifndef MOTUTAPU_CUDA_DENSEMATMUL_HPP
#define MOTUTAPU_CUDA_DENSEMATMUL_HPP

#include <Motutapu/compute/cuda/CudaParams.hpp>
#include <cuda_fp16.h>
#define __CUDACC__


namespace Motutapu::Cuda::Dense
{

__global__ void WmmaGemmHalf(half* Out, half* A, half* B,
                             size_t paddedColSizeA, size_t paddedColSizeB,
                             size_t paddedColSizeOut);

__global__ void WmmaGemmFloat(float* Out, half* A, half* B,
                              size_t paddedColSizeA, size_t paddedColSizeB,
                              size_t paddedColSizeOut);

__global__ void GemmHalf(half* out, half* A, half* B, size_t numRowOutA,
                         size_t numRowOutB, size_t numColARowB,
                         size_t paddedColSizeA, size_t paddedColSizeB,
                         size_t paddedColSizeOut, size_t size);

}

#endif
