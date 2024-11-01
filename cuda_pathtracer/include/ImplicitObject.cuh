#pragma once

#include "Vec3.cuh"
#include "Color.cuh"
#include "Ray.cuh"
#include "Intersection.cuh"

// Base class for all implicit objects
class ImplicitObject_t {
protected:
    Color_t color_;
    bool isEmissive_;
    Color_t emissionColor_;
    float emissionStrength_;

public:
    __host__ __device__ explicit ImplicitObject_t(const Color_t& color)
        : color_(color)
        , isEmissive_(false)
        , emissionColor_()
        , emissionStrength_(0.0f)
    {}

    __host__ __device__ virtual ~ImplicitObject_t() = default;

    // Pure virtual methods
    __host__ __device__ virtual Intersection_t intersect(const Ray_t& ray) const = 0;
    __host__ __device__ virtual Vec3f_t getNormalAt(const Point3f_t& point) const = 0;

    // Common methods
    __host__ __device__ void makeEmissive(const Color_t& emission_color, float strength) {
        isEmissive_ = true;
        emissionColor_ = emission_color;
        emissionStrength_ = strength;
    }

    __host__ __device__ bool isEmissive() const { return isEmissive_; }
    __host__ __device__ const Color_t& getEmissionColor() const { return emissionColor_; }
    __host__ __device__ float getEmissionStrength() const { return emissionStrength_; }
    __host__ __device__ const Color_t& getColor() const { return color_; }

    // Add virtual methods for light sampling
    __host__ __device__ virtual Point3f_t getCenter() const {
        return Point3f_t(); // Default implementation
    }

    __host__ __device__ virtual float getRadius() const {
        return 0.0f; // Default implementation
    }

    __host__ __device__ virtual bool isSphere() const {
        return false; // Default implementation
    }

    // Add a method to sample points on the surface
    __host__ __device__ virtual Point3f_t sampleSurface(float u, float v) const {
        return Point3f_t(); // Default implementation
    }
};

// Sphere implementation
class Sphere_t final : public ImplicitObject_t {
private:
    Point3f_t center_;
    float radius_;
    float radiusSquared_;

public:
    __host__ __device__ Sphere_t(const Point3f_t& center, float radius, const Color_t& color)
        : ImplicitObject_t(color)
        , center_(center)
        , radius_(radius)
        , radiusSquared_(radius * radius)
    {}

    __host__ __device__ Intersection_t intersect(const Ray_t& ray) const override {
        Intersection_t hit;
        hit.hit = false;

        const Vec3f_t oc = ray.origin - center_;
        const float a = 1.0f;  // Optimized since ray direction is normalized
        const float half_b = dot(oc, ray.direction);
        const float c = dot(oc, oc) - radiusSquared_;
        const float discriminant = half_b * half_b - a * c;

        if (discriminant < 0) {
            return hit;
        }

        const float sqrtd = sqrtf(discriminant);
        float root = (-half_b - sqrtd) / a;

        if (!ray.isValidDistance(root)) {
            root = (-half_b + sqrtd) / a;
            if (!ray.isValidDistance(root)) {
                return hit;
            }
        }

        hit.hit = true;
        hit.distance = root;
        hit.point = ray.at(root);
        hit.normal = getNormalAt(hit.point);
        hit.color = color_;
        hit.setFaceNormal(ray, hit.normal);

        if (isEmissive_) {
            hit.emission = emissionColor_ * emissionStrength_;
        }

        return hit;
    }

    __host__ __device__ Vec3f_t getNormalAt(const Point3f_t& point) const override {
        return (point - center_) / radius_;
    }

    __host__ __device__ Point3f_t getCenter() const override {
        return center_;
    }

    __host__ __device__ float getRadius() const override {
        return radius_;
    }

    __host__ __device__ bool isSphere() const override {
        return true;
    }
    
    __host__ __device__ Point3f_t sampleSurface(float u, float v) const override {
        const float theta = 2.0f * M_PI * u;
        const float phi = acosf(2.0f * v - 1.0f);
        
        const float sin_phi = sinf(phi);
        const Vec3f_t direction(
            cosf(theta) * sin_phi,
            sinf(theta) * sin_phi,
            cosf(phi)
        );
        
        return center_ + direction * radius_;
    }

    __host__ __device__ bool contains(const Point3f_t& point) const {
        return (point - center_).length_squared() <= radiusSquared_;
    }

    __host__ __device__ float getSurfaceArea() const {
        return 4.0f * M_PI * radiusSquared_;
    }
};

// Plane implementation
class Plane_t final : public ImplicitObject_t {
private:
    Point3f_t point_;
    Vec3f_t normal_;

public:
    __host__ __device__ Plane_t(const Point3f_t& point, const Vec3f_t& normal, const Color_t& color)
        : ImplicitObject_t(color)
        , point_(point)
        , normal_(normal.normalized())
    {}

    __host__ __device__ Intersection_t intersect(const Ray_t& ray) const override {
        Intersection_t hit;
        hit.hit = false;

        const float denom = dot(normal_, ray.direction);
        
        if (fabsf(denom) < 1e-8f) {
            return hit;
        }

        const float t = dot(point_ - ray.origin, normal_) / denom;
        
        if (!ray.isValidDistance(t)) {
            return hit;
        }

        hit.hit = true;
        hit.distance = t;
        hit.point = ray.at(t);
        hit.normal = normal_;
        hit.color = color_;
        hit.setFaceNormal(ray, normal_);

        if (isEmissive_) {
            hit.emission = emissionColor_ * emissionStrength_;
        }

        return hit;
    }

    __host__ __device__ Vec3f_t getNormalAt(const Point3f_t&) const override {
        return normal_;
    }

    __host__ __device__ const Point3f_t& getPoint() const { return point_; }
    __host__ __device__ const Vec3f_t& getNormal() const { return normal_; }
}; 