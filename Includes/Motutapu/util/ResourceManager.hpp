// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef MOTUTAPU_RESOURCEMANAGER_DECL_HPP
#define MOTUTAPU_RESOURCEMANAGER_DECL_HPP

#include <Motutapu/util/TensorDataDecl.hpp>
#include <Motutapu/util/ConcurrentQueue.hpp>
#include <list>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include <thread>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace Motutapu
{
struct Unit
{
    std::unordered_map<std::string, int> OutputMap;
    std::unordered_map<std::string, int> InputMap;

    std::unordered_map<std::string, int> InternalTensor;
};

struct Trajectory
{
    std::list<int> IdList;
};

struct WorkloadTracker
{
    std::unordered_map<int, Unit> UnitMap;
    std::shared_mutex m_mtx;
};

struct TensorPool
{
    std::unordered_map<int, Util::TensorData<float>> TensorDataMapFloat;
    std::unordered_map<int, Util::TensorData<int>> TensorDataMapInt;
    std::unordered_map<int, Util::TensorData<double>> TensorDataMapDouble;
    std::shared_mutex m_floatMapMtx;
    std::shared_mutex m_intMapMtx;
    std::shared_mutex m_doubleMapMtx;
};

struct ResourcePool
{
#ifdef WITH_CUDA
    Util::ConcurrentQueue<cudaStream_t> StreamPool;
#endif
    Util::ConcurrentQueue<std::thread> ThreadPool;
};
}

#endif