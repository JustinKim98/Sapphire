// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_POOL_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_POOL_CUH

#include <Sapphire/compute/dense/cuda/CudnnStruct.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void PoolForward2d(float* y, float* x, Shape4D xShape,
                            int windowHeight, int windowWidth, int strideRow,
                            int strideCol, int rowPadding, int columnPadding,
                            cudnnPoolingMode_t mode,
                            cudnnNanPropagation_t nanPropagation, int deviceId);

__host__ void PoolBackward2d(float* y, float* dy, float* x, float* dx,
                             Shape4D xShape, int windowHeight, int windowWidth,
                             int strideRow, int strideCol, int rowPadding,
                             int columnPadding, int deviceId);
}

#endif
