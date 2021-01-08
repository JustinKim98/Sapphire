// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_SPARSEMATRIX_DECL_HPP
#define MOTUTAPU_UTIL_SPARSEMATRIX_DECL_HPP

#if defined(__CUDACC__)  // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__)  // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)  // MSVC
#define ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for ALIGN macro for your host compiler!"
#endif

#include <cstdint>

template <typename T>
struct ALIGN(16) SparseMatrix
{
    uint32_t NumRows;
    uint32_t NNZ;
    uint32_t* ColIndex;
    uint32_t* RowIndex;
    T* V;
};

#define SPARSEMATRIX_PADDED_SIZE 32


#endif
