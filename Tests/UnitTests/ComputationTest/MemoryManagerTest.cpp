// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/util/MemoryManager.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include "doctest.h"

namespace Motutapu::Test
{
void hostAllocationTest()
{
    std::vector<unsigned int> size(100);

    std::random_device
        rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with
                             // rd()
    std::uniform_int_distribution<> distrib(500, 1000);

    size_t totalSize = 0;
    for (int i = 0; i < 100; i++)
    {
        size[i] = static_cast<unsigned int>(distrib(gen));
        totalSize += size[i];
        float* data = Util::MemoryManager::GetMemoryHost(size[i]);
        for (unsigned int j = 0; j < size[i]; j++)
        {
            data[j] = static_cast<float>(i);
        }
        Util::MemoryManager::UnAssignMemoryHost(data);
    }

    CHECK_EQ(Util::MemoryManager::GetTotalAllocationByteSizeHost(), totalSize);

    for (int i = 0; i < 100; i++)
    {
        float* data = Util::MemoryManager::GetMemoryHost(size[i]);
        for (unsigned int j = 0; j < size[i]; j++)
        {
            data[j] = static_cast<float>(i);
        }
        Util::MemoryManager::UnAssignMemoryHost(data);
    }

    Util::MemoryManager::ClearUnusedHostMemoryPool();
    CHECK_EQ(Util::MemoryManager::GetTotalAllocationByteSizeHost(), 0);
}

void cudaAllocationTest()
{
    std::vector<unsigned int> size(100);
    std::vector<float*> referenceDataVector(100);

    std::random_device
        rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with
    // rd()
    std::uniform_int_distribution<> distrib(500, 1000);

    size_t totalSize = 0;
    for (int i = 0; i < 100; i++)
    {
        size[i] = static_cast<unsigned int>(distrib(gen));
        totalSize += size[i];
        float* data = Util::MemoryManager::GetMemoryCuda(size[i], 0);

        Util::MemoryManager::UnAssignMemoryCuda(data, 0);
    }

    CHECK_EQ(Util::MemoryManager::GetTotalAllocationByteSizeCuda(), totalSize);

    for (int i = 0; i < 100; i++)
    {
        float* data = Util::MemoryManager::GetMemoryCuda(size[i], 0);

        Util::MemoryManager::UnAssignMemoryCuda(data, 0);
    }

    Util::MemoryManager::ClearUnusedCudaMemoryPool();
    CHECK_EQ(Util::MemoryManager::GetTotalAllocationByteSizeCuda(), 0);
}
}  // namespace Motutapu::Test