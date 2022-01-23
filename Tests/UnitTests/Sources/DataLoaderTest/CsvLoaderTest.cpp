// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/util/DataLoader/CsvLoader.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <DataLoaderTest/CsvLoaderTest.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/util/DeviceInfo.hpp>
#include <iostream>

namespace Sapphire::Test
{
void CsvLoaderTest(std::filesystem::path filePath, bool print)
{
    Util::CsvLoader<int> dataLoader(filePath);

    const DeviceInfo gpu(0, "cuda0");

    const Tensor labelTensor(Shape({ 1 }), gpu, Type::Dense, true);
    const Tensor dataTensor(Shape({ 28 * 28 }), gpu, Type::Dense, true);

    dataLoader.LoadData(dataTensor, 3, 1, 784);
    dataLoader.LoadData(labelTensor, 3, 0, 0);

    const std::vector<float> data = dataTensor.GetData();
    const std::vector<float> label = labelTensor.GetData();
    if (print)
    {
        for (const auto elem : data)
            std::cout << std::to_string(elem) << ", " << std::endl;
        std::cout << "label : " << std::to_string(label[0]) << std::endl;
    }

    Util::ResourceManager::ClearAll();
}
}
