// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_SPARSEMATRIX_HPP
#define Sapphire_SPARSEMATRIX_HPP

#if defined(__CUDACC__)  // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__)  // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)  // MSVC
#define ALIGN(n) __declspec(align(n))
#else
#error "Unsupported compiler"
#endif

#define MAX_NNZ_PER_ROW (1 << 10)
#define INF (~0)
#define DELETED_MARKER (~1)
#define THREADS_PER_BLOCK 128

#include <cstdint>

struct ALIGN(16) SparseMatrix
{
    float* V;
    uint32_t* COL;
    uint32_t* ROW;
    uint32_t NNZ;
    uint32_t M;
    uint32_t N;
    //! Padding bits to ensure this struct to be 48 bytes
    uint32_t Padding[3];
};

struct ALIGN(16) LoadDistMatrix
{
    uint32_t* Load;
    uint32_t* COL;
    uint32_t* ROW;
    uint32_t NNZ;
    uint32_t M;
    uint32_t N;
    //! Padding bits to ensure this struct to be 48 bytes
    uint32_t Padding[3];
};
#endif  // Sapphire_SPARSEMATRIX_HPP