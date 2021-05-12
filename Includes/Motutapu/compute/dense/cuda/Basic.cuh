// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CUDA_BASIC_CUH
#define MOTUTAPU_COMPUTE_CUDA_BASIC_CUH

#include <Motutapu/compute/cudaUtil/CudaParams.cuh>

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
__host__ void Dot(unsigned int totalSize, float* output, const float* inputA,
                  const float* inputB, unsigned int inputStride,
                  bool broadcastInputA, bool broadcastInputB);

//! out = input*scaleFactor
__host__ void Scale(float* output, const float* input, const float scaleFactor,
                    unsigned int totalSize);

//! out = input^T
__host__ void Transpose(float* output, const float* input,
                        unsigned int inputNumRows, unsigned int inputNumCols,
                        unsigned int batchSize, bool broadcastInput);
//! out = pow(input, factor)
__host__ void Pow(float* output, const float* input, const float factor,
                  unsigned int totalSize);

__host__ void cos(float* output, const float* input, unsigned int totalSize);

__host__ void sin(float* output, const float* input, unsigned int totalSize);

__host__ void tan(float* output, const float* input, unsigned int totalSize);

__host__ void cosh(float* output, const float* input, unsigned int totalSize);

__host__ void sinh(float* output, const float* input, unsigned int totalSize);

__host__ void tanh(float* output, const float* input, unsigned int totalSize);

__host__ void log(float* output, const float* input, unsigned int totalSize);

__host__ void log10(float* output, const float* input, unsigned int totalSize);

__host__ void ReLU(float* output, const float* input, unsigned int totalSize);

__host__ void ReLUDerivative(float* output, const float* input,
                             unsigned int totalSize);

__host__ void LeakyReLU(float* output, const float* input, float a,
                        unsigned int totalSize);

__host__ void LeakyReLUDerivative(float* output, const float* input, float a,
                                  unsigned int totalSize);

__host__ void Inverse(float* output, const float* input,
                      unsigned int totalSize);

__host__ void Mean(float* output, const float* input, unsigned int totalSize,
                   unsigned int unitSize);

__host__ void Softmax(float* output, const float* input, unsigned int totalSize,
                      unsigned int unitSize);

__host__ void SoftmaxBack(float* dx, const float* dy, const float* x,
                          unsigned int totalSize, unsigned int unitSize);
}  // namespace Motutapu::Compute::Cuda::Dense

#endif  // MOTUTAPU_BASIC_CUH
