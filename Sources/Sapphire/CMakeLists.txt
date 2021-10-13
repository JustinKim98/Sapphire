# Target name
set(target sapphire)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(cuda_targets)

# Includes
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries
)

# Sources
file(GLOB header_dir
        ${root_dir}/Includes)

file(GLOB_RECURSE headers
        ${header_dir}/*.hpp)

file(GLOB_RECURSE sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file(GLOB_RECURSE cuda_sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
 add_library(${target} ${sources})

if (USE_CUDA)
    include(../../CMake/IncludeCuda.cmake)
endif ()

find_package(OpenMP REQUIRED)

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        )

# Compile options

target_compile_options(${target}
        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}

        INTERFACE
        )

target_link_libraries(${target}
        PRIVATE
       ${cuda_targets}
        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        ${DEFAULT_LIBRARIES}
        OpenMP::OpenMP_CXX

        INTERFACE
        )

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include) 