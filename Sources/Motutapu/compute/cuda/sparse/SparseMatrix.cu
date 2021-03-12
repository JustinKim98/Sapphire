// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/CudaParams.cuh>
#include <Motutapu/compute/cuda/sparse/SparseMatrix.cuh>

namespace Motutapu::Compute
{
__host__ void DeepAllocateSparseMatrix(SparseMatrix* cudaTarget, size_t m,
                                         size_t nnz)
{
    cudaMalloc(&cudaTarget->V, sizeof(float) * nnz);
    cudaMalloc(&cudaTarget->COL, sizeof(size_t) * nnz);
    cudaMalloc(&cudaTarget->ROW, sizeof(size_t) * m);
    cudaTarget->M = m;
    cudaTarget->NNZ = nnz;
}

__host__ void ShallowAllocateSparseMatrix(SparseMatrix* target)
{
    cudaMalloc(target, sizeof(SparseMatrix));
}

__host__ void DeepFreeSparseMatrix(SparseMatrix* target)
{
    cudaFree(&dest->COL);
    cudaFree(&dest->ROW);
    cudaFree(&dest);
}

__host__ void ShallowFreeSparseMatrix(SparseMatrix* target)
{
    cudaFree(&dest->V);
    cudaFree(&dest->COL);
    cudaFree(&dest->ROW);
}

}  // namespace Motutapu::Compute