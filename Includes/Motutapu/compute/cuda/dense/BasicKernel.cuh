// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_BASICKERNEL_CUH
#define MOTUTAPU_BASICKERNEL_CUH

#define TILE_DIM 8

namespace Motutapu::Compute::Cuda::Dense
{
__global__ void AddKernel(float* output, const float* inputA,
                          const float* inputB, unsigned int offset,
                          unsigned int launchSize, unsigned int totalSize,
                          unsigned int inputStride, bool broadcastInputA,
                          bool broadcastInputB);

__global__ void AddKernelShared(float* output, const float* inputA,
                                const float* inputB, unsigned int offset,
                                unsigned int launchSize, unsigned int totalSize,
                                unsigned int inputStride, unsigned int numLoops,
                                bool broadcastInputA, bool broadcastInputB);

__global__ void AddKernelBroadcast(float* output, const float* inputA,
                                   const float* inputB, unsigned int offset,
                                   unsigned int totalSize,
                                   unsigned int inputStride,
                                   bool broadcastInputA, bool broadcastInputB);

__global__ void SubKernel(float* output, const float* inputA,
                          const float* inputB, unsigned int offset,
                          unsigned int launchSize, unsigned int totalSize,
                          unsigned int inputStride, bool broadcastInputA,
                          bool broadcastInputB);

__global__ void DotKernel(float* output, const float* inputA,
                          const float* inputB, unsigned int offset,
                          unsigned int launchSize, unsigned int totalSize,
                          unsigned int inputStride, bool broadcastInputA,
                          bool broadcastInputB);

__global__ void ScaleKernel(float* output, const float* input,
                            const float scaleFactor, unsigned int totalSize);

__global__ void TransposeKernel(float* output, const float* input,
                                unsigned int inputNumRows,
                                unsigned int inputNumCols, bool broadcastInput);

}  // namespace Motutapu::Compute::Cuda::Dense

#endif  // MOTUTAPU_BASICKERNEL_CUH
