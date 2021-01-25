// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_COMPUTE_HPP
#define MOTUTAPU_COMPUTE_COMPUTE_HPP

#include <Motutapu/compute/ComputeDecl.hpp>
#include <Motutapu/compute/cuda/dense/Gemm.cuh>

#include <Motutapu/tensor/TensorData.hpp>

namespace Motutapu::Compute
{
template <typename T>
void Gemm(Util::TensorData<T>& out, const Util::TensorData<T>& a,
          const Util::TensorData<T>& b)
{

}
}


#endif
