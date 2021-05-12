// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TEST_SPARSE_MEMORY_TEST_HPP
#define MOTUTAPU_TEST_SPARSE_MEMORY_TEST_HPP

#include <doctest.h>
#include <Motutapu/compute/sparse/Sparse.hpp>
#include <Motutapu/util/MemoryManager.hpp>
#include <random>

namespace Motutapu::Test
{
void GenerateFixedSparseArray(SparseMatrix** sparseMatrixArray, uint32_t m,
                              uint32_t n, uint32_t numMatrices);

void GenerateRandomSparseArray(SparseMatrix** sparseMatrixArray, uint32_t m,
                               uint32_t n, uint32_t numMatrices);
void SparseMemoryAllocationHost();

void LoadDistMemoryAllocationHost();

void SparseMemoryAllocationDevice();

void SparseMemoryCopyDeviceToDevice();

}  // namespace Motutapu::Test

#endif  // MOTUTAPU_SPARSE_HPP
