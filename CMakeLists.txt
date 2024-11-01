cmake_minimum_required(VERSION 3.10)
project(PathTracers VERSION 1.0.0 LANGUAGES CXX CUDA)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(X11 REQUIRED)
find_package(CUDAToolkit 12.0 REQUIRED)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")

# Add the two independent projects
add_subdirectory(cpu_pathtracer)
add_subdirectory(cuda_pathtracer)
