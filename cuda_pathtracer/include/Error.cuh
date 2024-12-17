#pragma once
#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << "\n" \
                      << "Error code: " << error << "\n" \
                      << "Error text: " << cudaGetErrorString(error) << "\n"; \
            cudaDeviceReset(); \
            throw cudaGetErrorString(error); \
        } \
    } while(0)
