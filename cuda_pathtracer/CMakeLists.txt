project(CUDAPathTracer CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -O3 \
    -use_fast_math \
    --expt-relaxed-constexpr \
    --extended-lambda \
    --relocatable-device-code=true \
    -lineinfo \
    --ptxas-options=-v \
    --maxrregcount=64")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
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
