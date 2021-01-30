#ifdef WITH_CUDA

#include <iostream>
#include <Motutapu/Test.hpp>
#include <cuda_runtime.h>
#include "doctest.h"

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
}
#endif