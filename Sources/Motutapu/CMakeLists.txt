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


if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    enable_language("CUDA")
    
    CUDA_ADD_LIBRARY(${target} ${sources} ${cuda_sources})
    message(STATUS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_property(TARGET motutapu PROPERTY CUDA_ARCHITECTURES 52 61 75)
    message(STATUS "CUDA enabled")

else()
    add_library(${target} ${sources})
endif()

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

#if(NOT USE_CUDA)

    target_link_libraries(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
	    ${DEFAULT_LIBRARIES}
    

        INTERFACE
    )
#endif()

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include) 