#include "Random.cuh"

// Implement the kernel here
__global__ void setupRandStates(curandState* rand_state, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        // Each thread gets different seed, same sequence
        curand_init(1984 + idx, 0, 0, &rand_state[idx]);
    }
} 