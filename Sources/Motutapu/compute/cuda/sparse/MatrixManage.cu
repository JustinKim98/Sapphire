//// Copyright (c) 2021, Justin Kim
//
//// We are making my contributions/submissions to this project solely in our
//// personal capacity and are not conveying any rights to any intellectual
//// property of any third parties.
//
//#include <Motutapu/compute/cuda/CudaParams.cuh>
//#include <Motutapu/compute/cuda/sparse/MatrixManage.cuh>
//#include <Motutapu/util/SparseMatrix.hpp>
//#include <cstdint>
//
//namespace Motutapu::Compute::Cuda::Sparse
//{
//__device__ void DeepAllocateSparseMatrix(SparseMatrix* dest)
//{
//    const auto nnz = dest->NNZ;
//    const auto rowArraySize = dest->NumRows + 1;
//    dest->ColIndex = static_cast<uint32_t*>(malloc(nnz * sizeof(uint32_t)));
//    dest->RowIndex =
//        static_cast<uint32_t*>(malloc(rowArraySize * sizeof(uint32_t)));
//    dest->V = static_cast<float*>(malloc(nnz * sizeof(float)));
//}
//
//__device__ void DeepFreeSparseMatrix(SparseMatrix* target)
//{
//    free(static_cast<void*>(target->ColIndex));
//    free(static_cast<void*>(target->RowIndex));
//    free(static_cast<void*>(target->V));
//}
//
//__device__ void ShallowFreeSparseMatrix(SparseMatrix* target)
//{
//    free(static_cast<void*>(target));
//}
//
//__global__ void DeepCopySparseMatrix(SparseMatrix* dest, SparseMatrix* src,
//                                     uint32_t rowOffset)
//{
//    const uint32_t numRows = src->NumRows;
//    const uint32_t* srcRowArray = src->RowIndex;
//    uint32_t* destRowArray = dest->RowIndex + rowOffset;
//    const uint32_t* srcColArray = src->ColIndex;
//    uint32_t* destColArray = dest->ColIndex;
//    const float* srcValue = src->V;
//    float* destValue = dest->V;
//
//    const auto totalNumWorkers = gridDim.x * blockDim.x;
//    auto index = blockDim.x * blockIdx.x + threadIdx.x;
//
//    while (index < numRows)
//    {
//        destRowArray[index] = srcRowArray[index];
//        auto cur = srcRowArray[index];
//        auto to = srcRowArray[index + 1];
//
//        while (cur < to)
//        {
//            destColArray[cur] = srcColArray[cur];
//            destValue[cur] = srcValue[cur];
//            cur++;
//        }
//        index += totalNumWorkers;
//    }
//}
//
//__global__ void DeepFreeKernel(SparseMatrix* targetArray, uint32_t size)
//{
//    const auto totalNumWorkers = gridDim.x * blockDim.x;
//    auto index = blockDim.x * blockIdx.x + threadIdx.x;
//    while (index < size)
//
//    {
//        SparseMatrix* destMatrix = targetArray + index;
//        DeepFreeSparseMatrix(destMatrix);
//
//        index += totalNumWorkers;
//    }
//}
//
//__global__ void DeepAllocateKernel(SparseMatrix* targetArray, uint32_t size)
//{
//    const auto totalNumWorkers = gridDim.x * blockDim.x;
//    auto index = blockDim.x * blockIdx.x + threadIdx.x;
//    while (index < size)
//    {
//        SparseMatrix* destMatrix = targetArray + index;
//        DeepAllocateSparseMatrix(destMatrix);
//
//        index += totalNumWorkers;
//    }
//}
//
//__global__ void DeepCopySparseMatrixOnGpu(SparseMatrix* destArray,
//                                          SparseMatrix* srcArray,
//                                          uint32_t size)
//{
//    const auto totalNumWorkers = gridDim.x * blockDim.x;
//    auto index = blockDim.x * blockIdx.x + threadIdx.x;
//
//    while (index < size)
//    {
//        SparseMatrix* destMatrix = destArray + index;
//        SparseMatrix* srcMatrix = srcArray + index;
//        DeepFreeSparseMatrix(destMatrix);
//
//        destMatrix->NNZ = srcMatrix->NNZ;
//        destMatrix->NumRows = srcMatrix->NumRows;
//        DeepAllocateSparseMatrix(destMatrix);
//
//        uint32_t numCopied = 0;
//        if (destMatrix->NumRows > DEFAULT_DIM_X)
//        {
//            DeepCopySparseMatrix
//                <<<destMatrix->NumRows / DEFAULT_DIM_X, DEFAULT_DIM_X>>>(
//                    destMatrix, srcMatrix, 0);
//            numCopied += (destMatrix->NumRows / DEFAULT_DIM_X) * DEFAULT_DIM_X;
//        }
//
//        if (destMatrix->NumRows % DEFAULT_DIM_X > 0)
//            DeepCopySparseMatrix<<<1, destMatrix->NumRows % DEFAULT_DIM_X>>>(
//                destMatrix, srcMatrix, numCopied);
//
//        index += totalNumWorkers;
//    }
//}
//
//__host__ void DeepCopyHostToGpu(SparseMatrix* deviceArray,
//                                SparseMatrix* hostArray, uint32_t size)
//{
//    for (size_t i = 0; i < size; ++i)
//    {
//        SparseMatrix* curDestPtr = deviceArray + i;
//        SparseMatrix* curSrcPtr = hostArray + i;
//
//        cudaMemcpyAsync(curDestPtr->RowIndex, curSrcPtr->RowIndex,
//                        (curSrcPtr->NumRows + 1) * sizeof(uint32_t),
//                        cudaMemcpyHostToDevice);
//        cudaMemcpyAsync(curDestPtr->ColIndex, curSrcPtr->ColIndex,
//                        (curSrcPtr->NNZ) * sizeof(uint32_t),
//                        cudaMemcpyHostToDevice);
//        cudaMemcpyAsync(curDestPtr->V, curSrcPtr->V,
//                        (curSrcPtr->NNZ) * sizeof(float),
//                        cudaMemcpyHostToDevice);
//    }
//}
//
//__host__ void DeepCopyGpuToHost(SparseMatrix* deviceArray,
//                                SparseMatrix* hostArray, uint32_t size)
//{
//    for (size_t i = 0; i < size; ++i)
//    {
//        SparseMatrix* curDestPtr = deviceArray + i;
//        SparseMatrix* curSrcPtr = hostArray + i;
//
//        cudaMemcpy(curDestPtr->RowIndex, curSrcPtr->RowIndex,
//                   (curSrcPtr->NumRows + 1) * sizeof(uint32_t),
//                   cudaMemcpyDeviceToHost);
//        cudaMemcpy(curDestPtr->ColIndex, curSrcPtr->ColIndex,
//                   (curSrcPtr->NNZ) * sizeof(uint32_t),
//                   cudaMemcpyDeviceToHost);
//        cudaMemcpy(curDestPtr->V, curSrcPtr->V,
//                   (curSrcPtr->NNZ) * sizeof(float), cudaMemcpyDeviceToHost);
//    }
//}
//
//__host__ void ShallowAllocateGpu(SparseMatrix* targetArray, uint32_t size)
//{
//    cudaMalloc(reinterpret_cast<void**>(&targetArray),
//               size * SPARSEMATRIX_PADDED_SIZE);
//}
//
//__host__ void DeepAllocateGpu(SparseMatrix* targetArray,
//                              SparseMatrix* hostRefArray, uint32_t size)
//{
//    cudaMemcpy(reinterpret_cast<void**>(&targetArray),
//               reinterpret_cast<void**>(&hostRefArray),
//               size * SPARSEMATRIX_PADDED_SIZE, cudaMemcpyHostToDevice);
//
//    uint32_t idx = 0;
//    uint32_t streamIdx = 0;
//    for (; idx < size; idx += DEFAULT_DIM_X, streamIdx++)
//    {
//        DeepAllocateKernel<<<1, DEFAULT_DIM_X>>>(
//            targetArray + idx, DEFAULT_DIM_X);
//    }
//
//    if (idx > 0)
//        idx -= DEFAULT_DIM_X;
//
//    DeepAllocateKernel<<<1, size - idx>>>(targetArray + idx, size - idx);
//}
//} // namespace Motutapu::Compute::Cuda::Sparse
