#ifdef WITH_CUDA

#include <Motutapu/compute/cuda/Memory.cuh>
#include <iostream>
#include "doctest.h"

namespace Motutapu::Test
{
void PrintCudaVersion()
{
    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;

    float* ptr;
    auto error = cudaMalloc((void**)&ptr, sizeof(float));
    CHECK(error == cudaSuccess);
    error = cudaFree(ptr);
    CHECK(error == cudaSuccess);

    std::cout << "Cuda Malloc and free successful" << std::endl;
}

void MallocTest()
{
    float* ptr;
    if (!Compute::Cuda::CudaMalloc(&ptr, 100))
        throw std::runtime_error("CudaMalloc failed");

    if (!Compute::Cuda::CudaFree(ptr))
    {
        throw std::runtime_error("CudaFree failed");
    }
}

}  // namespace Motutapu::Test

#endif