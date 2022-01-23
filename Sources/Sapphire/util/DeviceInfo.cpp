// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/util/DeviceInfo.hpp>
#include <stdexcept>


namespace Sapphire
{
DeviceInfo::DeviceInfo(int id, std::string name)
    : m_id(id),
      m_name(std::move(name))
{
#ifdef WITH_CUDA
    if (id >= GetAvailableCudaDeviceCount())
    {
        throw std::runtime_error("Cuda device has not been detected");
    }
    int majorCapability;
    int minorCapability;
    cudaDeviceGetAttribute(&majorCapability, cudaDevAttrComputeCapabilityMajor,
                           m_id);
    cudaDeviceGetAttribute(&minorCapability, cudaDevAttrComputeCapabilityMinor,
                           m_id);
    m_cudaCapability = majorCapability * 10 + minorCapability;
#endif
}

bool DeviceInfo::operator==(const DeviceInfo& device) const
{
    return m_id == device.m_id;
}

bool DeviceInfo::operator!=(const DeviceInfo& device) const
{
    return !(*this == device);
}
} // namespace Sapphire
