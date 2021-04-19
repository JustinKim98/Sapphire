// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_SPARSEMATRIX_HPP
#define MOTUTAPU_SPARSEMATRIX_HPP

#include <cstdint>

struct __attribute__((__packed__)) SparseMatrix
{
    uint32_t NNZ;
    uint32_t M;
    uint32_t N;
    float* V;
    uint32_t* COL;
    uint32_t* ROW;
};

struct __attribute__((__packed__)) LoadDistMatrix
{
    uint32_t NNZ;
    uint32_t M;
    uint32_t N;
    uint32_t* Load;
    uint32_t* COL;
    uint32_t* ROW;
};
#endif  // MOTUTAPU_SPARSEMATRIX_HPP