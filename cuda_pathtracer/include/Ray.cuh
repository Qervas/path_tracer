#pragma once

#include <cuda_runtime.h>
#include "Vec3.cuh"
#include "Color.cuh"
#include <cstdint>

class Ray_t {
public:
    // Ray types for optimization and debugging
    enum class Type {
        PRIMARY,    // Camera rays
        REFLECTED,  // Specular reflection
        REFRACTED,  // Transmission/refraction
        SHADOW,     // Shadow testing
        DIFFUSE     // Diffuse bounce
    };

    Vec3f_t origin;
    Vec3f_t direction;
    Color_t color;
    float importance;
    Type type;
    uint32_t depth;
    float min_t;
    float max_t;
    bool has_endpoint;
    Vec3f_t endpoint;

    // Constructor for new primary rays
    __host__ __device__ Ray_t()
        : origin()
        , direction()
        , color(1.0f, 1.0f, 1.0f)
        , importance(1.0f)
        , type(Type::PRIMARY)
        , depth(0)
        , min_t(1e-4f)
        , max_t(1e4f)
        , has_endpoint(false)
    {}
    
    __host__ __device__ Ray_t(const Vec3f_t& o, const Vec3f_t& d, Type ray_type = Type::PRIMARY) 
        : origin(o)
        , direction(d.normalized())
        , color(1.0f, 1.0f, 1.0f)
        , importance(1.0f)
        , type(ray_type)
        , depth(0)
        , min_t(1e-4f)
        , max_t(1e4f)
        , has_endpoint(false)
    {}

    // Constructor for secondary rays (bounces)
    __host__ __device__ Ray_t(
        const Vec3f_t& o,
        const Vec3f_t& d,
        const Color_t& c,
        float imp,
        Type t,
        uint32_t dep)
        : origin(o)
        , direction(d.normalized())
        , color(c)
        , importance(imp)
        , type(t)
        , depth(dep)
        , min_t(1e-4f)
        , max_t(1e4f)
        , has_endpoint(false)
    {}

    // Factory methods for creating new rays
    __host__ __device__ Ray_t createReflectedRay(
        const Vec3f_t& new_origin,
        const Vec3f_t& new_direction,
        const Color_t& attenuation) const {
        return Ray_t(
            new_origin,
            new_direction,
            color * attenuation,
            importance * 0.8f,
            Type::REFLECTED,
            depth + 1
        );
    }

    __host__ __device__ Ray_t createRefractedRay(
        const Vec3f_t& new_origin,
        const Vec3f_t& new_direction,
        const Color_t& attenuation) const {
        return Ray_t(
            new_origin,
            new_direction,
            color * attenuation,
            importance * 0.8f,
            Type::REFRACTED,
            depth + 1
        );
    }

    __host__ __device__ Ray_t createDiffuseRay(
        const Vec3f_t& new_origin,
        const Vec3f_t& new_direction,
        const Color_t& attenuation) const {
        return Ray_t(
            new_origin,
            new_direction,
            color * attenuation,
            importance * 0.5f,
            Type::DIFFUSE,
            depth + 1
        );
    }

    __host__ __device__ Ray_t createShadowRay(
        const Vec3f_t& new_origin,
        const Vec3f_t& target) const {
        Ray_t shadow_ray(
            new_origin,
            (target - new_origin).normalized(),
            Color_t(1.0f, 1.0f, 1.0f),
            1.0f,
            Type::SHADOW,
            depth
        );
        shadow_ray.setEndpoint(target);
        return shadow_ray;
    }

    // Utility methods
    __host__ __device__ Vec3f_t at(float t) const {
        return origin + direction * t;
    }

    __host__ __device__ void setEndpoint(const Vec3f_t& end) {
        endpoint = end;
        max_t = (end - origin).length();
        has_endpoint = true;
    }

    __host__ __device__ bool hasEndpoint() const {
        return has_endpoint;
    }

    __host__ __device__ const Vec3f_t& getEndpoint() const {
        return endpoint;
    }

    __host__ __device__ bool isValidDistance(float t) const {
        return t >= min_t && t <= max_t;
    }
};
