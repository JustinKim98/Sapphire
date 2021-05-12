# Target name
set(target sapphire)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

set(cudnn_lib
        libcudnn.so
        libcudnn.so.8
        libcudnn.so.8.1.0
        libcudnn_adv_infer.so
        libcudnn_adv_infer.so.8
        libcudnn_adv_infer.so.8.1.0
        libcudnn_adv_train.so
        libcudnn_adv_train.so.8
        libcudnn_adv_train.so.8.1.0
        libcudnn_cnn_infer.so
        libcudnn_cnn_infer.so.8
        libcudnn_cnn_infer.so.8.1.0
        libcudnn_cnn_train.so
        libcudnn_cnn_train.so.8
        libcudnn_cnn_train.so.8.1.0
        libcudnn_ops_infer.so
        libcudnn_ops_infer.so.8
        libcudnn_ops_infer.so.8.1.0
        libcudnn_ops_train.so
        libcudnn_ops_train.so.8
        libcudnn_ops_train.so.8.1.0
        )

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
    CUDA_ADD_LIBRARY(sapphire_cuda ${cuda_sources})
    set_property(TARGET sapphire_cuda PROPERTY CUDA_ARCHITECTURES 86)
    add_library(sapphire ${sources})
    find_library(sapphire_cublas NAMES libcublas.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_library(cudnn NAMES ${cudnn_lib} HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
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
        sapphire_cuda
        ${sapphire_cublas}
        ${cudnn}
        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        ${DEFAULT_LIBRARIES}
        OpenMP::OpenMP_CXX

        INTERFACE
        )

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include) 