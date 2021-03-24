// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_SPARSEMATRIX_HPP
#define MOTUTAPU_SPARSEMATRIX_HPP

#include <cstdint>

struct SparseMatrix
{
    float* V;
    uint32_t* COL;
    uint32_t* ROW;

    uint32_t NNZ;
    uint32_t M;
};

#endif  // MOTUTAPU_SPARSEMATRIX_HPP