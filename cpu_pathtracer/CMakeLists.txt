project(CPUPathTracer CXX)

# C++ settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    add_compile_options(
        /W4 /O2 /Oi /GL /MP /arch:AVX2
        /fp:fast  # Fast floating-point model
    )
    add_link_options(/LTCG)
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -O3 -march=native -ffast-math -flto
        -fopenmp  # OpenMP support for parallel processing
    )
endif()

# Source files
file(GLOB_RECURSE CPP_SOURCES 
    "src/*.cpp"
    "src/*.hpp"
)

# Main executable
add_executable(${PROJECT_NAME} ${CPP_SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        X11
)

if(NOT MSVC)
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
            gomp  # OpenMP library
    )
endif()
