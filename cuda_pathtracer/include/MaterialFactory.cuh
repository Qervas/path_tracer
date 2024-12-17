#pragma once

#include "Material.cuh"
#include "Color.cuh"
#include "Error.cuh"

// Device kernels
__global__ void createLambertianKernel(Material_t* ptr, Color_t albedo) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(ptr) Lambertian_t(albedo);
    }
}

__global__ void createEmissiveKernel(Material_t* ptr, Color_t color, float strength) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(ptr) Emissive_t(color, strength);
    }
}

__global__ void createDielectricKernel(Material_t* ptr, float ior) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(ptr) Dielectric_t(ior);
    }
}

__global__ void createMetalKernel(Material_t* ptr, Color_t albedo, float roughness) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(ptr) Metal_t(albedo, roughness);
    }
}

__global__ void createGlossyKernel(Material_t* ptr, Color_t albedo, float roughness, float metallic) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(ptr) Glossy_t(albedo, roughness, metallic);
    }
}

// Host-side factory class
class MaterialFactory {
public:
    static __host__ void createMaterialsOnDevice(
        Material_t** d_white_diffuse,
        Material_t** d_red_diffuse,
        Material_t** d_green_diffuse,
        Material_t** d_light,
        Material_t** d_glass,
        Material_t** d_metal,
		Material_t** d_glossy,
		Material_t** d_mirror
    ) {
        // Allocate device memory for materials
        CUDA_CHECK(cudaMalloc(d_white_diffuse, sizeof(Lambertian_t)));
        CUDA_CHECK(cudaMalloc(d_red_diffuse, sizeof(Lambertian_t)));
        CUDA_CHECK(cudaMalloc(d_green_diffuse, sizeof(Lambertian_t)));
        CUDA_CHECK(cudaMalloc(d_light, sizeof(Emissive_t)));
        CUDA_CHECK(cudaMalloc(d_glass, sizeof(Dielectric_t)));
        CUDA_CHECK(cudaMalloc(d_metal, sizeof(Metal_t)));
	    CUDA_CHECK(cudaMalloc(d_glossy, sizeof(Glossy_t)));
        CUDA_CHECK(cudaMalloc(d_mirror, sizeof(Metal_t)));



        // Launch kernels to construct materials
        createLambertianKernel<<<1,1>>>(*d_white_diffuse, Color_t(0.73f));
        createLambertianKernel<<<1,1>>>(*d_red_diffuse, Color_t(0.65f, 0.05f, 0.05f));
        createLambertianKernel<<<1,1>>>(*d_green_diffuse, Color_t(0.12f, 0.45f, 0.15f));
        createEmissiveKernel<<<1,1>>>(*d_light, Color_t(1.0f), 15.0f);
        createDielectricKernel<<<1,1>>>(*d_glass, 1.5f);
        createMetalKernel<<<1,1>>>(*d_metal, Color_t(0.95f), 0.0f);
	    createGlossyKernel<<<1,1>>>(*d_glossy, Color_t(0.7f, 0.7f, 0.7f), 0.2f, 0.8f);
		createMetalKernel<<<1,1>>>(*d_mirror, Color_t(0.98f, 0.98f, 0.98f), 0.0f);


        CUDA_CHECK(cudaDeviceSynchronize());
    }

    static __host__ void cleanup(
        Material_t* d_white_diffuse,
        Material_t* d_red_diffuse,
        Material_t* d_green_diffuse,
        Material_t* d_light,
        Material_t* d_glass,
        Material_t* d_metal,
        Material_t* d_glossy,
        Material_t* d_mirror
    ) {
        if (d_white_diffuse) CUDA_CHECK(cudaFree(d_white_diffuse));
        if (d_red_diffuse) CUDA_CHECK(cudaFree(d_red_diffuse));
        if (d_green_diffuse) CUDA_CHECK(cudaFree(d_green_diffuse));
        if (d_light) CUDA_CHECK(cudaFree(d_light));
        if (d_glass) CUDA_CHECK(cudaFree(d_glass));
        if (d_metal) CUDA_CHECK(cudaFree(d_metal));
	    if (d_glossy) CUDA_CHECK(cudaFree(d_glossy));

    }
};
