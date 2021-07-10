
file(GLOB_RECURSE cuda_sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)


if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
set(cudnn_lib
        cudnn_adv_infer64_8.dll
        cudnn_adv_train64_8.dll
        cudnn_cnn_infer64_8.dll
        cudnn_cnn_train64_8.dll
        cudnn_ops_infer64_8.dll
        cudnn_ops_train64_8.dll
        cudnn64_8.dll
        )
else()

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

endif ()

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        message(STATUS "ROOT DIR:" ${CUDA_TOOLKIT_ROOT_DIR})
        CUDA_ADD_LIBRARY(sapphire_cuda ${cuda_sources})
        set_property(TARGET sapphire_cuda PROPERTY CUDA_ARCHITECTURES 86)
        SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
        find_library(sapphire_cublas NAMES cublas64_11.dll cublasLt_11.dll HINTS ${CUDA_TOOLKIT_ROOT_DIR}/bin)
        find_library(sapphire_cusparse NAMES cusparse64_11.dll HINTS ${CUDA_TOOLKIT_ROOT_DIR}/bin)
        find_library(cudnn NAMES ${cudnn_lib} HINTS ${CUDA_TOOLKIT_ROOT_DIR}/bin)
    else()
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        CUDA_ADD_LIBRARY(sapphire_cuda ${cuda_sources})
        set_property(TARGET sapphire_cuda PROPERTY CUDA_ARCHITECTURES 86)
        find_library(sapphire_cublas NAMES libcublas.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        find_library(sapphire_cusparse NAMES libcusparse.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        find_library(cudnn NAMES ${cudnn_lib} HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    endif()


    set(cuda_targets
        sapphire_cuda
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        cudnn)