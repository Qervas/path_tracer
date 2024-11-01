#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "Vec3.cuh"
#include "Color.cuh"
#include "Camera.cuh"
#include "Scene.cuh"

// Device structures
struct GPUCamera {
    float3 origin;
    float3 forward;
    float3 right;
    float3 up;
    float fov;
    int width;
    int height;
};

struct GPUSphere {
    float3 center;
    float radius;
    float3 color;
    float3 emission;
    bool is_emissive;
};

struct GPUPlane {
    float3 point;
    float3 normal;
    float3 color;
};

// Declare external constants
extern __constant__ GPUCamera d_camera;
extern __constant__ GPUSphere d_spheres[16];
extern __constant__ GPUPlane d_planes[16];
extern __constant__ int d_num_spheres;
extern __constant__ int d_num_planes;

// Kernel declaration
extern "C" void launchRenderKernel(float4* output, uint32_t width, uint32_t height, uint32_t frame_count, dim3 grid, dim3 block);

// Host functions for initialization
void initializeGPUData(const Camera_t& camera, const Scene_t* scene); 