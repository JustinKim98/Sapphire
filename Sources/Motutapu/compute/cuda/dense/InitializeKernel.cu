//// Copyright (c) 2021, Jaewoo Kim
//
//// We are making my contributions/submissions to this project solely in our
//// personal capacity and are not conveying any rights to any intellectual
//// property of any third parties.
//
//#include <Motutapu/compute/cuda/dense/InitializeKernel.cuh>
//#include <Motutapu/compute/cuda/CudaParams.hpp>
//
//namespace Motutapu::Compute::Cuda::Dense
//{
//__global__ void initRandomKernel(curandState* state)
//{
//    const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
//    /* Each thread gets same seed, a different sequence
//       number, no offset */
//    curand_init(1234, id, 0, &state[id]);
//}
//
//__global__ void NormalFloatKernel(float* data, float mean, float sd,
//                                  unsigned int size,
//                                  curandState* state)
//{
//    const unsigned int id = threadIdx.x;
//
//    const auto numLoopPerThread = blockDim.x == 0
//                                      ? size / blockDim.x
//                                      : size / blockDim.x + 1;
//
//    curandState localState = state[id];
//
//    for (unsigned int i = id * numLoopPerThread; i < size; i++)
//    {
//        data[i] = (curand_normal(&localState) - mean) / sd;
//    }
//}
//
//__global__ void NormalHalfKernel(half* data, half mean, half sd,
//                                 unsigned int size,
//                                 curandState* state)
//{
//    const unsigned int id = threadIdx.x;
//
//    const auto numLoopPerThread =
//        blockDim.x == 0 ? size / blockDim.x : size / blockDim.x + 1;
//
//    curandState localState = state[id];
//
//    for (unsigned int i = id * numLoopPerThread; i < size; i++)
//    {
//        data[i] = (__float2half(curand_normal(&localState)) - mean) / sd;
//    }
//}
//
//__global__ void ScalarFloatKernel(float* data, float value, unsigned int size)
//{
//    const unsigned int id = threadIdx.x;
//
//    const auto numLoopPerThread =
//        blockDim.x == 0 ? size / blockDim.x : size / blockDim.x + 1;
//
//    for (unsigned int i = id * numLoopPerThread; i < size; i++)
//    {
//        data[i] = value;
//    }
//}
//
//__global__ void ScalarHalfKernel(half* data, half value, unsigned int size)
//{
//    const unsigned int id = threadIdx.x;
//
//    const auto numLoopPerThread =
//        blockDim.x == 0 ? size / blockDim.x : size / blockDim.x + 1;
//
//    for (unsigned int i = id * numLoopPerThread; i < size; i++)
//    {
//        data[i] = value;
//    }
//}
//} // namespace Motutapu::Compute::Cuda::Dense
