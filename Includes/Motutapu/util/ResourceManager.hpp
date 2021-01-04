// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef MOTUTAPU_RESOURCEMANAGER_DECL_HPP
#define MOTUTAPU_RESOURCEMANAGER_DECL_HPP

#include <functional>
#include <Motutapu/util/TensorDataDecl.hpp>
#include <Motutapu/util/ConcurrentQueue.hpp>
#include <list>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include <thread>

#include <Motutapu/tensor/TensorDecl.hpp>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace Motutapu
{
template <typename T>
class Unit
{
protected:
    std::unordered_map<std::string, int> OutputMap;
    std::unordered_map<std::string, int> InputMap;

    std::unordered_map<std::string, int> InternalVariableTensor;
    std::unordered_map<std::string, int> InternalConstantTensor;

    std::unordered_map<std::string, std::string> StringLiterals;
    std::unordered_map<std::string, T> ScalarLiterals;
    std::unordered_map<std::string, int> IntegerLiterals;

    std::function<std::vector<Tensor<T>>(
        std::vector<Tensor<T>>)> BackwardFuncFloat;
};

class Trajectory
{
    std::list<int> IdList;
};

class TensorPool
{
public:
    Util::TensorData<float>& FloatTensorData(int tensorId);
    Util::TensorData<int>& IntTensorData(int tensorId);
    Util::TensorData<double>& DoubleTensorData(int tensorId);

private:
    std::unordered_map<int, Util::TensorData<float>> TensorDataMapFloat;
    std::unordered_map<int, Util::TensorData<int>> TensorDataMapInt;
    std::unordered_map<int, Util::TensorData<double>> TensorDataMapDouble;
    std::shared_mutex m_floatMapMtx;
    std::shared_mutex m_intMapMtx;
    std::shared_mutex m_doubleMapMtx;
};

class FunctionPool
{
protected:
    template <typename T>
    static void RegisterUnit(Unit<T>, int unitId);

    template <typename T>
    static Unit<T> GetUnit(int unitId);

private:
    std::unordered_map<int, Unit<float>>
    m_floatUnitMap;
    std::unordered_map<int, Unit<int>>
    m_intUnitMap;
    std::unordered_map<int, Unit<double>>
    m_doubleUnitMap;

    std::shared_mutex m_mtx;
};

class ResourcePool
{
public:
    static std::thread PopThread();
    static void PushThread(std::thread& thread);

private:
    static Util::ConcurrentQueue<std::thread> ThreadPool;

#ifdef WITH_CUDA
public:
    static cudaStream_t PopStream();
    static void PushStream();
private:
    static Util::ConcurrentQueue<cudaStream_t> m_streamPool;

#endif
};
}

#endif
