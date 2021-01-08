// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CUDA_MEMORY_CUH
#define MOTUTAPU_CUDA_MEMORY_CUH
#include <Motutapu/util/SparseMatrixDecl.hpp>
#include <Motutapu/compute/cuda/CudaParams.hpp>
//#ifdef WITH_CUDA

__host__ bool CudaSetDevice(int deviceId);

template <typename T>
__global__ void CopyOnGpu(T* dest, const T* const src, unsigned int size)
{
    const auto index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size)
    {
        dest[index] = src[index];
    }
}



template <typename T>
__host__ __device__ bool CudaMalloc(T** ptr, size_t size)
{
    const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(ptr),
                                         size * sizeof(T));
    return error == cudaSuccess;
}

template <typename T>
__host__ __device__ bool CudaFree(T** ptr)
{
    const cudaError_t error = cudaFree(reinterpret_cast<void**>(ptr));
    return error == cudaSuccess;
}

template <typename T>
__host__ bool MemcpyHostToGpu(T* gpuPtr, T* hostPtr, size_t size)
{
    const cudaError_t error =
        cudaMemcpy(reinterpret_cast<void*>(gpuPtr),
                   reinterpret_cast<void*>(hostPtr), size * sizeof(T),
                   cudaMemcpyHostToDevice);

    return error == cudaSuccess;
}

template <typename T>
__host__ bool MemcpyGpuToHost(T* hostPtr, T* gpuPtr, size_t size)
{
    const cudaError_t error =
        cudaMemcpy(reinterpret_cast<void*>(hostPtr),
                   reinterpret_cast<void*>(gpuPtr),
                   size * sizeof(T),
                   cudaMemcpyDeviceToHost);

    return error == cudaSuccess;
}

template <typename T>
__host__ void MemcpyGpuToGpu(T* dest, T* src, size_t size)
{
    unsigned int elementsCopied = 0;

    if (size > MAX_THREAD_DIM_X)
    {
        cudaStream_t stream0;
        cudaStreamCreate(&stream0);
        const auto requiredBlocks = size / MAX_THREAD_DIM_X;
        CopyOnGpu<<<requiredBlocks, MAX_THREAD_DIM_X>>>(
            dest, src, requiredBlocks * MAX_THREAD_DIM_X);

        elementsCopied += requiredBlocks * MAX_THREAD_DIM_X;
    }

    CopyOnGpu<<<1, size>>>(dest + elementsCopied, src + elementsCopied,
                           size - elementsCopied);
}

template <typename T>
__host__ void CopySparseHostToGpu(SparseMatrix<T>* dest,
                                  SparseMatrix<T>* src,
                                  unsigned int batchSize)
{
    
}

template <typename T>
__host__ bool CopySparseGpuToHost(SparseMatrix<T>* dest,
                                  SparseMatrix<T>* src,
                                  unsigned int batchSize);

template <typename T>
__host__ void CopySparseGpuToGpu(SparseMatrix<T>* destArray,
                                 SparseMatrix<T>* srcArray,
                                 unsigned int batchSize)
{
    const auto destArrayTotalByteSize =
        (2 * destArray->NNZ + destArray->RowIndex + 1) * sizeof(T);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                       destArrayTotalByteSize * batchSize);

    unsigned int numCopied = 0;
    if (batchSize > MAX_THREAD_DIM_X)
    {
        CopySparseMatrixOnGpu<T>
        <<<batchSize / MAX_THREAD_DIM_X, MAX_THREAD_DIM_X>>>(
                destArray, srcArray, batchSize);
        numCopied += (batchSize / MAX_THREAD_DIM_X) * MAX_THREAD_DIM_X;
    }

    if (batchSize - numCopied > 0)
    {
        CopySparseMatrixOnGpu<T><<<1, batchSize - numCopied>>>(
            destArray + numCopied, srcArray + numCopied, batchSize);
    }


}

//#endif
#endif
