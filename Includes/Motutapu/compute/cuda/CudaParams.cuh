// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUDAPARAMS_HPP
#define CUDAPARAMS_HPP

#ifdef WITH_CUDA

#define MAX_THREAD_DIM_X 1024
#define MAX_THREAD_DIM_Y 1024
#define MAX_THREAD_DIM_Z 64
#define MAX_GRID_DIM 65535

#define DEFAULT_DIM_X 64
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <device_launch_parameters.h>
#include <cassert>

#endif

#endif