// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_BASIC_BACKWARD_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_BASIC_BACKWARD_CUH

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void DotBackward(unsigned int totalSize, float* da, float* db,
                          const float* dy, const float* a, const float* b,
                          unsigned inputStride, bool
                          broadcastInputA, bool broadcastInputB);

__host__ void PowBackward(unsigned int totalSize, float* dx, float* dy,
                          float* x);

__host__ void cosBackward(unsigned int totalSize, float* dx, float* dy,
                          float* x);

__host__ void sinBackward(unsigned int totalSize, float* dx, float* dy,
                          float* x);

__host__ void tanBackward(unsigned int totalSize, float* dx, float* dy,
                          float* x);

__host__ void coshBackward(unsigned int totalSize, float* dx, float* dy,
                           float* x);

__host__ void sinhBackward(unsigned int totalSize, float* dx, float* dy,
                           float* x);

__host__ void tanhBackward(unsigned int totalSize, float* dx, float* dy,
                           float* x);

__host__ void logBackward(unsigned int totalSize, float* dx, float* dy,
                          float* x);

__host__ void log10Backward(unsigned int totalSize, float* dx, float* dy,
                            float* x);

__host__ void ReLUBackward(unsigned int totalSize, float* dx, float* dy,
                           float* x);

__host__ void LeakyReluBackward(unsigned int totalSize, float* dx, float* dy,
                                float* x);

__host__ void InverseBackward(unsigned int totalSize, float* dx,
                              const float* dy, const float* x);

__host__ void MeanBackward(float* dx, const float* x, const float* dy,
                           unsigned int yTotalSize, unsigned int unitSize,
                           unsigned int stride);
}

#endif
