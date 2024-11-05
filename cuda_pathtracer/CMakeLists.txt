project(CUDAPathTracer CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# CUDA compiler flags - common flags
set(CUDA_COMMON_FLAGS "\
    --expt-relaxed-constexpr \
    --extended-lambda \
    --relocatable-device-code=true \
    --source-in-ptx \
    -lineinfo \
    --ptxas-options=-v \
    --maxrregcount=64")

# Set different flags for Debug and Release
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_COMMON_FLAGS} -G -g -O0")
    set(CMAKE_BUILD_TYPE Debug)
    add_compile_options(-g)
endif()

# Source files
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
	
# Main executable
add_executable(${PROJECT_NAME} ${CUDA_SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set CUDA specific properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 89
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        X11
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::curand
)

# Add these lines at the end
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Create cuda-ptx directory
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cuda-ptx)
    
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>
        $<$<COMPILE_LANGUAGE:CUDA>:-src-in-ptx>
        $<$<COMPILE_LANGUAGE:CUDA>:-keep>
        $<$<COMPILE_LANGUAGE:CUDA>:-keep-dir ${CMAKE_CURRENT_BINARY_DIR}/cuda-ptx>
    )
endif()
