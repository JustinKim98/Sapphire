# CMake version
cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# Declare project
project(motutapu)

option(USE_CUDA "USE_CUDA" ON)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Set enable output of compile commands during generation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 17)
# Includes
include_directories(Includes)
include_directories(Libraries)
include_directories(Libraries/doctest/doctest)

# Compile options
include(CMake/CheckAVX.cmake)
include(CMake/CompileOptions.cmake)

# Build type - Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Overrides
set(CMAKE_MACOSX_RPATH ON)

add_definitions(-DUSE_CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
enable_language("CUDA")
#set(
#        CUDA_NVCC_FLAGS
#        ${CUDA_NVCC_FLAGS};
#        -arch=sm_80
#        -gencode=arch=compute_70,code=sm_70
#        -gencode=arch=compute_75,code=sm_75
#        -gencode=arch=compute_80,code=sm_80
#        -gencode=arch=compute_86,code=sm_86
#        -gencode=arch=compute_86,code=compute_86
#        --default-stream per-thread
#        --device-c
#        --cudart=static
#        --cudadevrt=static
#)

# Project modules
add_subdirectory(Libraries/doctest)
add_subdirectory(Sources/Motutapu)
add_subdirectory(Tests/UnitTests)

if(USE_CUDA)
    add_compile_definitions(WITH_CUDA)
endif()

# Code coverage - Debug only
# NOTE: Code coverage results with an optimized (non-Debug) build may be misleading
option(BUILD_COVERAGE "Build code coverage" OFF)
if (CMAKE_BUILD_TYPE MATCHES Debug AND CMAKE_COMPILER_IS_GNUCXX AND BUILD_COVERAGE)
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}_coverage UnitTests coverage)
endif()