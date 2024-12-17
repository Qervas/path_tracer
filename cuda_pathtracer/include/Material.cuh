#pragma once

#include "Vec3.cuh"
#include "Color.cuh"
#include "Ray.cuh"
#include "Intersection.cuh"
#include "ScatterRecord.cuh"
#include <curand_kernel.h>

// Base material class
class Material_t {
public:
    __device__ virtual ~Material_t() = default;
    
    __device__ virtual bool scatter(
        const Ray_t& ray_in,
        const Intersection_t& hit,
        ScatterRecord_t& srec,
        curandState* rand_state) const = 0;
    
    __device__ virtual float scatteringPdf(
        const Ray_t& ray_in,
        const Intersection_t& hit,
        const Ray_t& scattered) const {
        return 0.0f;
    }

    __device__ virtual Color_t emitted(
        const Ray_t& ray_in,
        const Intersection_t& hit) const {
        return Color_t(0.0f);
    }
};

// Lambertian (diffuse) material
class Lambertian_t : public Material_t {
private:
    Color_t albedo_;

public:
    __device__ Lambertian_t(const Color_t& albedo) : albedo_(albedo) {}

    __device__ virtual bool scatter(
        const Ray_t& ray_in,
        const Intersection_t& hit,
        ScatterRecord_t& srec,
        curandState* rand_state) const override {
        
        // Create ONB (orthonormal basis) from hit normal
        Vec3f_t w = hit.normal;
        Vec3f_t a = (fabsf(w.x) > 0.9f) ? Vec3f_t(0, 1, 0) : Vec3f_t(1, 0, 0);
        Vec3f_t v = cross(w, a).normalized();
        Vec3f_t u = cross(v, w);

        // Cosine weighted hemisphere sampling
        float r1 = 2.0f * M_PI * curand_uniform(rand_state);
        float r2 = curand_uniform(rand_state);
        float r2s = sqrtf(r2);

        Vec3f_t direction = (u * cosf(r1) * r2s +
                           v * sinf(r1) * r2s +
                           w * sqrtf(1.0f - r2)).normalized();

        srec.scattered_ray = Ray_t(hit.point + direction * 0.001f, direction);
        srec.attenuation = albedo_;
        srec.pdf = dot(hit.normal, direction) / M_PI;
        srec.is_specular = false;

        return true;
    }

    __device__ virtual float scatteringPdf(
        const Ray_t& ray_in,
        const Intersection_t& hit,
        const Ray_t& scattered) const override {
        float cosine = dot(hit.normal, scattered.direction);
        return cosine <= 0 ? 0 : cosine / M_PI;
    }
};

// Metal material
class Metal_t : public Material_t {
private:
    Color_t albedo_;
    float roughness_;

    __device__ Vec3f_t reflect(const Vec3f_t& v, const Vec3f_t& n) const {
        return v - n * 2.0f * dot(v, n);
    }

public:
    __device__ Metal_t(const Color_t& albedo, float roughness)
        : albedo_(albedo)
        , roughness_(roughness < 1.0f ? roughness : 1.0f)
    {}

    __device__ virtual bool scatter(
        const Ray_t& ray_in,
        const Intersection_t& hit,
        ScatterRecord_t& srec,
        curandState* rand_state) const override {
        
        Vec3f_t reflected = reflect(ray_in.direction, hit.normal);
        
        // randomness for roughness
        if (roughness_ > 0.0f) {
            Vec3f_t random = Vec3f_t(
                curand_uniform(rand_state) * 2.0f - 1.0f,
                curand_uniform(rand_state) * 2.0f - 1.0f,
                curand_uniform(rand_state) * 2.0f - 1.0f
            );
            reflected = (reflected + random * roughness_).normalized();
        }

        srec.scattered_ray = Ray_t(hit.point + reflected * 0.001f, reflected);
        srec.attenuation = albedo_;
        srec.is_specular = true;
        srec.pdf = 1.0f;

        return dot(reflected, hit.normal) > 0;
    }
};

// Dielectric (glass) material
class Dielectric_t : public Material_t {
private:
    float ior_;  // Index of refraction

    __device__ Vec3f_t reflect(const Vec3f_t& v, const Vec3f_t& n) const {
        return v - n * 2.0f * dot(v, n);
    }

    __device__ Vec3f_t refract(const Vec3f_t& uv, const Vec3f_t& n, float etai_over_etat) const {
        float cos_theta = fminf(dot(-uv, n), 1.0f);
        Vec3f_t r_out_perp = (uv + n * cos_theta) * etai_over_etat;
        Vec3f_t r_out_parallel = n * -sqrtf(fabsf(1.0f - r_out_perp.length_squared()));
        return r_out_perp + r_out_parallel;
    }

    __device__ float schlick(float cosine, float ref_idx) const {
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
    }

public:
    __device__ explicit Dielectric_t(float ior) : ior_(ior) {}

    __device__ virtual bool scatter(
        const Ray_t& ray_in,
        const Intersection_t& hit,
        ScatterRecord_t& srec,
        curandState* rand_state) const override {
        
        srec.is_specular = true;
        srec.pdf = 1.0f;
        srec.attenuation = Color_t(1.0f);

        float etai_over_etat = hit.frontFace ? (1.0f / ior_) : ior_;

        float cos_theta = fminf(dot(-ray_in.direction, hit.normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

        bool cannot_refract = etai_over_etat * sin_theta > 1.0f;
        Vec3f_t direction;

        if (cannot_refract || schlick(cos_theta, etai_over_etat) > curand_uniform(rand_state)) {
            direction = reflect(ray_in.direction, hit.normal);
        } else {
            direction = refract(ray_in.direction, hit.normal, etai_over_etat);
        }

        srec.scattered_ray = Ray_t(hit.point + direction * 0.001f, direction);
        return true;
    }
};

class Glossy_t : public Material_t {
private:
    Color_t albedo_;
    float roughness_;
    float metallic_;

public:
    __device__ Glossy_t(const Color_t& albedo, float roughness, float metallic = 1.0f)
        : albedo_(albedo)
        , roughness_(roughness)
        , metallic_(metallic)
    {}

    __device__ bool scatter(const Ray_t& ray_in, const Intersection_t& isect, ScatterRecord_t& srec, curandState* rand_state) const override {
        Vec3f_t reflected = reflect(ray_in.direction, isect.normal);

        // Add randomness based on roughness
        Vec3f_t random_vec = Random_t::randomUnitVector(rand_state);
        Vec3f_t scattered_direction = reflected + random_vec * roughness_;
        scattered_direction = scattered_direction.normalized();

        // Ensure the scattered ray is above the surface
        if (dot(scattered_direction, isect.normal) < 0) {
            scattered_direction = reflected;
        }

        srec.scattered_ray = Ray_t(isect.point, scattered_direction, Ray_t::Type::REFLECTED);

        // Interpolate between metallic reflection and diffuse reflection
        Color_t specular_color = metallic_ > 0.5f ? albedo_ : Color_t(1.0f);
        srec.attenuation = lerp(albedo_, specular_color, metallic_);

        srec.pdf = 1.0f;
        srec.is_specular = true;
        return true;
    }

    __device__ float scatteringPdf(const Ray_t& ray_in, const Intersection_t& isect, const Ray_t& scattered) const override {
        return 1.0f;
    }

private:
    __device__ static Vec3f_t reflect(const Vec3f_t& v, const Vec3f_t& n) {
        return v - 2.0f * dot(v, n) * n;
    }

    __device__ static Color_t lerp(const Color_t& a, const Color_t& b, float t) {
        return Color_t(
            a.r * (1.0f - t) + b.r * t,
            a.g * (1.0f - t) + b.g * t,
            a.b * (1.0f - t) + b.b * t
        );
    }
};


// Emissive material
class Emissive_t : public Material_t {
private:
    Color_t emission_;
    float strength_;

public:
    __device__ Emissive_t(const Color_t& emission, float strength)
        : emission_(emission)
        , strength_(strength)
    {}

    __device__ virtual bool scatter(
        const Ray_t& ray_in,
        const Intersection_t& hit,
        ScatterRecord_t& srec,
        curandState* rand_state) const override {
        return false;  // Emissive materials don't scatter light
    }

    __device__ virtual Color_t emitted(
        const Ray_t& ray_in,
        const Intersection_t& hit) const override {
        return hit.frontFace ? emission_ * strength_ : Color_t(0.0f);
    }
}; 