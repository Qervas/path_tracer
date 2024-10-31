cmake_minimum_required(VERSION 3.10)
project(RayTracer VERSION 1.0.0 LANGUAGES CXX)

# Compiler settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Advanced compiler flags for optimization and warnings
if(MSVC)
    add_compile_options(
        /W4     # Warning level 4
        /O2     # Maximum optimization
        /Oi     # Generate intrinsic functions
        /GL     # Whole program optimization
        /MP     # Multi-processor compilation
        /arch:AVX2  # Enable AVX2 instructions
    )
    add_link_options(/LTCG)  # Link-time code generation
else()
    add_compile_options(
        -Wall 
        -Wextra 
        -Wpedantic
        -O3                 # Maximum optimization
        -march=native      # Optimize for local machine
        -ffast-math        # Aggressive floating-point optimizations
        -flto              # Link-time optimization
    )
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Source files
file(GLOB_RECURSE SOURCES 
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.hpp"
)

# Main executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Enable testing
enable_testing()

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
