// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/util/ResourceManager.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include "doctest.h"

namespace Sapphire::Test
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
            Util::ResourceManager::GetMemoryHost(size[i] * sizeof(float)));
        for (unsigned int j = 0; j < size[i]; j++)
        {
            data[j] = static_cast<float>(i);
        }
        Util::ResourceManager::DeReferenceHost(static_cast<void*>(data));
    }

    CHECK_EQ(Util::ResourceManager::GetTotalByteSizeHost(), totalSize);

    for (int i = 0; i < 100; i++)
    {
        auto* data = static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(size[i] * sizeof(float)));
        for (unsigned int j = 0; j < size[i]; j++)
        {
            data[j] = static_cast<float>(i);
        }
        Util::ResourceManager::DeReferenceHost(static_cast<void*>(data));
    }

    Util::ResourceManager::ClearFreeHostMemoryPool();
    CHECK_EQ(Util::ResourceManager::GetTotalByteSizeHost(), 0);
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
            Util::ResourceManager::GetMemoryCuda(size[i] * sizeof(float), 0));

        Util::ResourceManager::DeReferenceCuda(static_cast<void*>(data), 0);
    }

    CHECK_EQ(Util::ResourceManager::GetTotalByteSizeCuda(), totalSize);

    for (int i = 0; i < 100; i++)
    {
        auto* data = static_cast<float*>(
            Util::ResourceManager::GetMemoryCuda(size[i] * sizeof(float), 0));
        Util::ResourceManager::DeReferenceCuda(static_cast<void*>(data), 0);
    }

    Util::ResourceManager::ClearFreeCudaMemoryPool();
    CHECK_EQ(Util::ResourceManager::GetTotalByteSizeCuda(), 0);
}
}  // namespace Sapphire::Test