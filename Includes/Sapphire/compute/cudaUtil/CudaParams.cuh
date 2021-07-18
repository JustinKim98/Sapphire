// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUDA_PARAMS_HPP
#define CUDA_PARAMS_HPP

#ifdef WITH_CUDA

#define MAX_THREAD_DIM_X 1024
#define MAX_THREAD_DIM_Y 1024
#define MAX_THREAD_DIM_Z 64
#define MAX_GRID_DIM 65535
#define NUM_LOOPS  8
#define DEFAULT_DIM_X 64
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cusparse.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            throw std::runtime_error("CUDA failed");                   \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            throw std::runtime_error("CUSPARSE failed");                   \
        }                                                                  \
    }

#endif

#endif