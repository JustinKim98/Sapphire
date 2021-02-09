// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_BASIC_CUH
#define MOTUTAPU_BASIC_CUH

#include <Motutapu/compute/cuda/CudaParams.cuh>

namespace Motutapu::Compute::Cuda::Dense
{
//! out = A + B
__host__ void Add(unsigned int totalSize, float* output, const float* inputA,
                  const float* inputB, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB);

//! out = A - B
__host__ void Sub(unsigned int totalSize, float* output, const float* inputA,
                  const float* inputB, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB);

//! out = inner_product(inputA, inputB)
__host__ void Dot(float* output, const float* inputA, const float* inputB,
                  unsigned int totalSize, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB);

//! out = input*scaleFactor
__host__ void Scale(float* output, const float* input, const float scaleFactor,
                    unsigned int totalSize, unsigned int inputStride,
                    bool broadcastInput);

//! out = input^T
__host__ void Transpose(float* output, const float* input,
                        unsigned int inputNumRows, unsigned int inputNumCols,
                        unsigned int batchSize, bool broadcastInput);
}  // namespace Motutapu::Compute::Cuda::Dense

#endif  // MOTUTAPU_BASIC_CUH
