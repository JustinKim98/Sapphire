// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_DEVICE_HPP
#define SAPPHIRE_DEVICE_HPP
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace Sapphire
{
enum class ComputeMode
{
    Host,
    Cuda,
};

class DeviceInfo
{
 public:
    DeviceInfo() = default;

    DeviceInfo(int id, std::string name);
    ~DeviceInfo() = default;

    DeviceInfo(const DeviceInfo& device) = default;
    DeviceInfo(DeviceInfo&& device) noexcept = default;
    DeviceInfo& operator=(const DeviceInfo& device) = default;
    DeviceInfo& operator=(DeviceInfo&& device) noexcept = default;

    bool operator==(const DeviceInfo& device) const;
    bool operator!=(const DeviceInfo& device) const;

    [[nodiscard]] std::string Name() const
    {
        return m_name;
    }

    [[nodiscard]] int GetID() const
    {
        return m_id;
    }

#ifdef WITH_CUDA

    [[nodiscard]] int GetCudaCapability() const
    {
        return m_cudaCapability;
    }

    static int GetAvailableCudaDeviceCount()
    {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }

#endif

 private:
    int m_id = -1;
    std::string m_name = "Undefined";

#ifdef WITH_CUDA
    int m_cudaCapability = 0;
#endif
};
}  // namespace Sapphire

#endif