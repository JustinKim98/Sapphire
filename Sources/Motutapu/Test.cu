//#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <iostream>

namespace Test::Cuda
{
void PrintCudaVersion()
{
    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;

    float* toMalloc;
    auto error = cudaMalloc((void**)&toMalloc, 100 * sizeof(float));
    if (error != cudaSuccess)
        throw std::runtime_error("Allocation failure");

    error = cudaFree(toMalloc);
    if (error != cudaSuccess)
        throw std::runtime_error("Free failure");

    std::cout << "CudaSuccess" << std::endl;
}
}  // namespace Test::Cuda
   //#endif