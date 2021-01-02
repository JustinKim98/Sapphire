// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TENSORDATA_DECL_HPP
#define MOTUTAPU_TENSORDATA_DECL_HPP

#include <Motutapu/util/DenseMatrix.hpp>
#include <Motutapu/util/SparseMatrix.hpp>
#include <Motutapu/util/ShapeDecl.hpp>

namespace Motutapu::Util
{
template <typename T>
struct TensorData
{
    DenseMatrix<T>* DenseMatrix;
    SparseMatrix<T>* SparseMatrix;
    Shape TensorShape;
    unsigned long DenseTotalLength;
    unsigned long SparseTotalLength;
    bool Mode; //True if Dense, False if Sparse
};

}

#endif