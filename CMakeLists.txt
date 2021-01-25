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

# Includes
include_directories(Includes)
include_directories(Impl)
include_directories(Libraries)
include_directories(Libraries/doctest/doctest)

# Build library
if(USE_CUDA)
    set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
	find_package(CUDA 11.0)
    set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
	if(NOT CUDA_FOUND)
		message(STATUS "CUDA not found")
		set(USE_CUDA OFF)
    else()
        add_compile_definitions(WITH_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
        message(STATUS "CUDA inculde path :  ${CUDA_INCLUDE_DIRS}")
        message(STATUS "CUDA found")
        set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}; 
        -arch=sm_80 
        -gencode=arch=compute_70,code=sm_70
        -gencode=arch=compute_75,code=sm_75
        -gencode=arch=compute_80,code=sm_80
        -gencode=arch=compute_86,code=sm_86
        -gencode=arch=compute_86,code=compute_86
        --std=c++17
        --default-stream per-thread
        --device-c
        )
	endif()
endif()

# Compile options
include(CMake/CheckAVX.cmake)
include(CMake/CompileOptions.cmake)

# Build type - Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Overrides
set(CMAKE_MACOSX_RPATH ON)

# Project modules
add_subdirectory(Libraries/doctest)
add_subdirectory(Sources/Motutapu)
add_subdirectory(Tests/UnitTests)

message(STATUS "FLAGS : ${CMAKE_NVCC_FLAGS}")

# Code coverage - Debug only
# NOTE: Code coverage results with an optimized (non-Debug) build may be misleading
option(BUILD_COVERAGE "Build code coverage" OFF)
if (CMAKE_BUILD_TYPE MATCHES Debug AND CMAKE_COMPILER_IS_GNUCXX AND BUILD_COVERAGE)
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}_coverage UnitTests coverage)
endif()