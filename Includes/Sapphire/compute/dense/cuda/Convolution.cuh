// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_CONVOLUTION_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_CONVOLUTION_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <Sapphire/compute/dense/cuda/CudnnStruct.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void CreateCudnnConv2DMetaData(CudnnConv2DMetaData* metaData,
                                      Shape4D xShape, Shape4D filterShape,
                                      int strideRow, int strideCol,
                                      int dilationRow, int dilationCol,
                                      int rowPadding, int columnPadding,
                                      int deviceId);

__host__ void Conv2DForward(float* y, float* x,
                            float* filter, Shape4D inputShape,
                            Shape4D filterShape,
                            int strideRow, int strideCol, int dilationRow,
                            int dilationCol, int rowPadding, int columnPadding,
                            int deviceId);

__host__ void Conv2DBackward(float* dx, float* filter,
                             float* dFilter, float* x,
                             float* dy, Shape4D inputShape,
                             Shape4D filterShape, int strideRow, int strideCol,
                             int dilationRow, int dilationCol, int rowPadding,
                             int columnPadding, int deviceId);
} // namespace Sapphire::Compute::Cuda

#endif  // Sapphire_CONVOLUTION_CUH
