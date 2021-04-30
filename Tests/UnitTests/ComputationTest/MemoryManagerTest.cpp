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
        auto* data = static_cast<float*>(
            Util::MemoryManager::GetMemoryHost(size[i] * sizeof(float)));
        for (unsigned int j = 0; j < size[i]; j++)
        {
            data[j] = static_cast<float>(i);
        }
        Util::MemoryManager::DeReferenceHost(static_cast<void*>(data));
    }

    CHECK_EQ(Util::MemoryManager::GetTotalByteSizeHost(), totalSize);

    for (int i = 0; i < 100; i++)
    {
        auto* data = static_cast<float*>(
            Util::MemoryManager::GetMemoryHost(size[i] * sizeof(float)));
        for (unsigned int j = 0; j < size[i]; j++)
        {
            data[j] = static_cast<float>(i);
        }
        Util::MemoryManager::DeReferenceHost(static_cast<void*>(data));
    }

    Util::MemoryManager::ClearUnusedHostMemoryPool();
    CHECK_EQ(Util::MemoryManager::GetTotalByteSizeHost(), 0);
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
        auto* data = static_cast<float*>(
            Util::MemoryManager::GetMemoryCuda(size[i] * sizeof(float), 0));

        Util::MemoryManager::DeReferenceCuda(static_cast<void*>(data), 0);
    }

    CHECK_EQ(Util::MemoryManager::GetTotalByteSizeCuda(), totalSize);

    for (int i = 0; i < 100; i++)
    {
        auto* data = static_cast<float*>(
            Util::MemoryManager::GetMemoryCuda(size[i] * sizeof(float), 0));
        Util::MemoryManager::DeReferenceCuda(static_cast<void*>(data), 0);
    }

    Util::MemoryManager::ClearUnusedCudaMemoryPool();
    CHECK_EQ(Util::MemoryManager::GetTotalByteSizeCuda(), 0);
}
}  // namespace Motutapu::Test