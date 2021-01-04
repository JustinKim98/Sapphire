// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TENSORDATA_DECL_HPP
#define MOTUTAPU_TENSORDATA_DECL_HPP

#include <Motutapu/util/DenseMatrix.hpp>
#include <Motutapu/util/SparseMatrix.hpp>
#include <Motutapu/tensor/Shape.hpp>

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
    unsigned long BatchSize;

    bool Mode; // True if Dense, False if Sparse

    static TensorData* CreateTensorData(unsigned long batchSize, Shape shape,
                                        bool mode);
    static TensorData* DestroyTensorData(TensorData<T>& tensorData);

    static void DenseToSparse(TensorData<T>& tensorData);
    static void SparseToDense(TensorData<T>& tensorData);

    static void CopyTensorData(TensorData<T>& dest, const TensorData<T>& src);
    static void CopyCPUToGPU(TensorData<T>& dest, TensorData<T>& src);
    static void CopyGPUToCPU(TensorData<T>& dest, TensorData<T>& src);
};
}

#endif
