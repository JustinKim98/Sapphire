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
    #set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
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
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17 --gpu-architecture=sm_50")
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


# Code coverage - Debug only
# NOTE: Code coverage results with an optimized (non-Debug) build may be misleading
option(BUILD_COVERAGE "Build code coverage" OFF)
if (CMAKE_BUILD_TYPE MATCHES Debug AND CMAKE_COMPILER_IS_GNUCXX AND BUILD_COVERAGE)
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}_coverage UnitTests coverage)
endif()