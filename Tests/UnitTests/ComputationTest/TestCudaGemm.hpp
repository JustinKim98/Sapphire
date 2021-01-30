// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_TEST_TESTCUDAGEMM_HPP
#define MOTUTAPU_TEST_TESTCUDAGEMM_HPP

#include <Motutapu/compute/cuda/Memory.cuh>
#include <Motutapu/compute/cuda/dense/Gemm.cuh>
#include <Motutapu/tensor/Tensor.hpp>
#include <Motutapu/tensor/TensorData.hpp>

namespace Motutapu::Test
{
void TensorGemmTest();
}

#endif
