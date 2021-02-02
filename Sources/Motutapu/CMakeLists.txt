# Target name
set(target motutapu)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

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

if (USE_CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    CUDA_ADD_LIBRARY(motutapu_cuda ${cuda_sources})
    #CUDA_ADD_CUBLAS_TO_TARGET(motutapu_cuda)
    set_property(TARGET motutapu_cuda PROPERTY CUDA_ARCHITECTURES 86)
    add_library(motutapu ${sources})
else ()
    add_library(${target} ${sources})
endif ()

find_package(OpenMP REQUIRED)

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        )

# Compile options
target_compile_options(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}

        INTERFACE
        )

target_link_libraries(${target}
        PRIVATE
        motutapu_cuda
        cublas
        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        ${DEFAULT_LIBRARIES}
        OpenMP::OpenMP_CXX

        INTERFACE
        )

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include) 