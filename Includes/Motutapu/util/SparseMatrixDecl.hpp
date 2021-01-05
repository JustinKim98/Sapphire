// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_SPARSEMATRIX_DECL_HPP
#define MOTUTAPU_UTIL_SPARSEMATRIX_DECL_HPP


template <typename T>
struct SparseMatrix
{
    T* V;
    unsigned int* ColIndex;
    unsigned int* RowIndex;

    unsigned int NumRows;
    unsigned int NumCols;
};


#endif
