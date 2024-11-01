#pragma once

#include <cuda_runtime.h>

// CUDA-compatible Vec3 class
template<typename T>
class Vec3_t {
public:
    T x, y, z;

    // Constructors
    __host__ __device__ Vec3_t() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3_t(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    __host__ __device__ Vec3_t(const Vec3_t& v) = default;

    // Basic operations
    __host__ __device__ Vec3_t operator-() const {
        return Vec3_t(-x, -y, -z);
    }

    __host__ __device__ Vec3_t& operator+=(const Vec3_t& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    __host__ __device__ Vec3_t& operator-=(const Vec3_t& v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
	}

    __host__ __device__ Vec3_t& operator*=(T t) {
        x *= t; y *= t; z *= t;
        return *this;
    }

    __host__ __device__ Vec3_t& operator/=(T t) {
        return *this *= (1/t);
    }

    // Length operations
    __host__ __device__ T length_squared() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ T length() const {
        return sqrtf(length_squared());
    }

    __host__ __device__ Vec3_t& normalize() {
        T len = length();
        if (len > 0) {
            *this *= (1 / len);
        }
        return *this;
    }

    __host__ __device__ Vec3_t normalized() const {
        Vec3_t result = *this;
        result.normalize();
        return result;
    }

    __host__ __device__ float3 toFloat3() const {
        return make_float3(x, y, z);
    }

    __host__ __device__ static Vec3_t<T> fromFloat3(const float3& f) {
        return Vec3_t<T>(f.x, f.y, f.z);
    }
};

// Type aliases
using Vec3f_t = Vec3_t<float>;
using Point3f_t = Vec3_t<float>;

// Vector operations
template<typename T>
__host__ __device__ inline Vec3_t<T> operator+(const Vec3_t<T>& u, const Vec3_t<T>& v) {
    return Vec3_t<T>(u.x + v.x, u.y + v.y, u.z + v.z);
}

template<typename T>
__host__ __device__ inline Vec3_t<T> operator-(const Vec3_t<T>& u, const Vec3_t<T>& v) {
    return Vec3_t<T>(u.x - v.x, u.y - v.y, u.z - v.z);
}

template<typename T>
__host__ __device__ inline Vec3_t<T> operator*(const Vec3_t<T>& v, T t) {
    return Vec3_t<T>(v.x * t, v.y * t, v.z * t);
}

template<typename T>
__host__ __device__ inline Vec3_t<T> operator*(T t, const Vec3_t<T>& v) {
    return v * t;
}

template<typename T>
__host__ __device__ inline Vec3_t<T> operator/(const Vec3_t<T>& v, T t) {
    return v * (1/t);
}

template<typename T>
__host__ __device__ inline T dot(const Vec3_t<T>& u, const Vec3_t<T>& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

template<typename T>
__host__ __device__ inline Vec3_t<T> cross(const Vec3_t<T>& u, const Vec3_t<T>& v) {
    return Vec3_t<T>(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

