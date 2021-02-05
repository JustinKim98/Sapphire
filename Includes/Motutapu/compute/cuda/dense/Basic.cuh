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
__host__ void Add(float* out, float* A, float* B);
//! out = out + A
__host__ void Add(float* out, float* A);
//! out = A - B
__host__ void Sub(float* out, float* A, float* B);
//! out = out - A
__host__ void Sub(float* out, float* A);

__host__ void Scale(float* out, float* A, float factor);

__host__ void Transpose(float* out, float* A, float factor);
}

#endif  // MOTUTAPU_BASIC_CUH
