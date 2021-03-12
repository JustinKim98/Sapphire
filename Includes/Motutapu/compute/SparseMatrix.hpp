// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_SPARSEMATRIX_HPP
#define MOTUTAPU_SPARSEMATRIX_HPP

#include <cstdlib>

namespace Motutapu::Compute
{
struct SparseMatrix
{
    float* V;
    size_t* COL;
    size_t* ROW;
    size_t M;
    size_t NNZ;
};
}  // namespace Motutapu::Compute

#endif  // MOTUTAPU_SPARSEMATRIX_HPP
