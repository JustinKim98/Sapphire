// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_BASIC_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_BASIC_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
//! out = A + B
__host__ void Add(unsigned int totalSize, float* y, const float* a,
                  const float* b, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB);

//! out = A - B
__host__ void Sub(unsigned int totalSize, float* y, const float* a,
                  const float* b, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB);

//! out = inner_product(a, b)
__host__ void Dot(unsigned int totalSize, float* y, const float* a,
                  const float* b, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB);

//! out = x*scaleFactor
__host__ void Scale(float* y, const float* x, const float scaleFactor,
                    unsigned int totalSize);

//! out = x^T
__host__ void Transpose(float* y, const float* x,
                        unsigned int inputNumRows, unsigned int inputNumCols,
                        unsigned int batchSize, bool broadcastInput);
//! out = pow(x, factor)
__host__ void Pow(float* y, const float* x, const float factor,
                  unsigned int totalSize);

__host__ void log(float* y, const float* x, unsigned int totalSize);

__host__ void log10(float* y, const float* x, unsigned int totalSize);

__host__ void ReLU(float* y, const float* x, unsigned int totalSize);

__host__ void ReLUDerivative(float* y, const float* x,
                             unsigned int totalSize);

__host__ void LeakyReLU(float* y, const float* x, float a,
                        unsigned int totalSize);

__host__ void LeakyReLUBackward(float* y, const float* x, float a,
                                  unsigned int totalSize);

__host__ void Inverse(float* y, const float* x,
                      unsigned int totalSize);

__host__ void Mean(float* y, const float* x, unsigned int totalSize,
                   unsigned int unitSize);

__host__ void Softmax(float* y, const float* x, unsigned int totalSize,
                      unsigned int unitSize);

__host__ void SoftmaxBack(float* dx, const float* dy, const float* x,
                          unsigned int totalSize, unsigned int unitSize);
}  // namespace Sapphire::Compute::Cuda::Dense

#endif  // Sapphire_BASIC_CUH
