#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include "Vec3.cuh"

// Only declare the kernel
__global__ void setupRandStates(curandState* rand_state, int num_pixels);

class Random_t {
private:
    curandState* rand_state;

public:
    __host__ static void initCurand(curandState* rand_state, int width, int height) {
        int num_pixels = width * height;
        // Setup random number generator states for each pixel
        setupRandStates<<<(num_pixels + 255) / 256, 256>>>(rand_state, num_pixels);
    }

    __device__ static float get(curandState* local_rand_state) {
        return curand_uniform(local_rand_state);
    }

    __device__ static float get(curandState* local_rand_state, float min, float max) {
        return min + (max - min) * get(local_rand_state);
    }

    __device__ static Vec3f_t randomInUnitDisk(curandState* local_rand_state) {
        float r = sqrtf(get(local_rand_state));  // sqrt for uniform distribution
        float theta = 2.0f * M_PI * get(local_rand_state);
        return Vec3f_t(r * cosf(theta), r * sinf(theta), 0.0f);
    }

    __device__ static Vec3f_t randomInUnitSphere(curandState* local_rand_state) {
        float z = get(local_rand_state, -1.0f, 1.0f);
        float t = get(local_rand_state, 0.0f, 2.0f * M_PI);
        float r = sqrtf(1.0f - z * z);
        return Vec3f_t(r * cosf(t), r * sinf(t), z);
    }

    __device__ static Vec3f_t randomUnitVector(curandState* local_rand_state) {
        return randomInUnitSphere(local_rand_state).normalized();
    }

    __device__ static Vec3f_t randomCosineDirection(curandState* local_rand_state) {
        float r1 = get(local_rand_state);
        float r2 = get(local_rand_state);
        float z = sqrtf(1.0f - r2);
        float phi = 2.0f * M_PI * r1;
        float x = cosf(phi) * sqrtf(r2);
        float y = sinf(phi) * sqrtf(r2);
        return Vec3f_t(x, y, z);
    }
};

// Host-side random number generation
class HostRandom_t {
private:
    static std::mt19937 generator;
    static std::uniform_real_distribution<float> distribution;

public:
    static void init();
    static float get();
    static float get(float min, float max);
    static Vec3f_t randomInUnitDisk();
};