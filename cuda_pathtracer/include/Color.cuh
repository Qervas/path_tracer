#pragma once

#include <cuda_runtime.h>
#include "Vec3.cuh"

// CUDA-compatible Color class
class Color_t {
public:
    float r, g, b;

    // Constructors
    __host__ __device__ Color_t() : r(0), g(0), b(0) {}
	__host__ __device__ Color_t(float t) : r(t), g(t), b(t) {}
    __host__ __device__ Color_t(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
    __host__ __device__ Color_t(const Vec3f_t& v) : r(v.x), g(v.y), b(v.z) {}

    // Operators
    __host__ __device__ Color_t& operator+=(const Color_t& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    __host__ __device__ Color_t& operator*=(const Color_t& other) {
        r *= other.r;
        g *= other.g;
        b *= other.b;
        return *this;
    }

    __host__ __device__ Color_t& operator*=(float t) {
        r *= t;
        g *= t;
        b *= t;
        return *this;
    }

	__host__ __device__ Color_t& operator=(float3 f) {
		r = f.x;
		g = f.y;
		b = f.z;
		return *this;
	}

    // Utility functions
    __host__ __device__ Color_t clamp() const {
        return Color_t(
            fminf(fmaxf(r, 0.0f), 1.0f),
            fminf(fmaxf(g, 0.0f), 1.0f),
            fminf(fmaxf(b, 0.0f), 1.0f)
        );
    }

    __host__ __device__ float4 toFloat4() const {
        return make_float4(r, g, b, 1.0f);
    }

    __host__ __device__ static Color_t fromFloat3(const float3& f) {
        return Color_t(f.x, f.y, f.z);
    }	
};

// Non-member operators
__host__ __device__ inline Color_t operator+(const Color_t& a, const Color_t& b) {
    return Color_t(a.r + b.r, a.g + b.g, a.b + b.b);
}

// add with float3
__host__ __device__ inline Color_t operator+(const Color_t& a, const float3& b) {
	return Color_t(a.r + b.x, a.g + b.y, a.b + b.z);
}

__host__ __device__ inline Color_t operator*(const Color_t& a, const Color_t& b) {
    return Color_t(a.r * b.r, a.g * b.g, a.b * b.b);
}

__host__ __device__ inline Color_t operator*(const Color_t& c, float t) {
    return Color_t(c.r * t, c.g * t, c.b * t);
}

__host__ __device__ inline Color_t operator*(float t, const Color_t& c) {
    return c * t;
}

// Common colors as device functions
namespace Colors {
    __device__ __host__ inline Color_t Black() { return Color_t(0.0f, 0.0f, 0.0f); }
    __device__ __host__ inline Color_t White() { return Color_t(1.0f, 1.0f, 1.0f); }
    __device__ __host__ inline Color_t Red()   { return Color_t(1.0f, 0.0f, 0.0f); }
    __device__ __host__ inline Color_t Green() { return Color_t(0.0f, 1.0f, 0.0f); }
    __device__ __host__ inline Color_t Blue()  { return Color_t(0.0f, 0.0f, 1.0f); }
}

// Alternative: Define colors as constant arrays that can be used in device code
namespace GPUColors {
    __device__ __constant__ const float3 BLACK = {0.0f, 0.0f, 0.0f};
    __device__ __constant__ const float3 WHITE = {1.0f, 1.0f, 1.0f};
    __device__ __constant__ const float3 RED   = {1.0f, 0.0f, 0.0f};
    __device__ __constant__ const float3 GREEN = {0.0f, 1.0f, 0.0f};
    __device__ __constant__ const float3 BLUE  = {0.0f, 0.0f, 1.0f};

    __device__ __host__ inline Color_t fromFloat3(const float3& f) {
        return Color_t(f.x, f.y, f.z);
    }
}
