#pragma once

#include "Vec3.cuh"
#include "Color.cuh"
#include "Ray.cuh"
#include "Intersection.cuh"

// Base class for all polygons
class Polygon_t {
protected:
    Color_t color_;
    Vec3f_t normal_;
    bool normalsDirty_;  // Flag for lazy normal computation

public:
    __host__ __device__ explicit Polygon_t(const Color_t& color)
        : color_(color), normalsDirty_(true) {}

    __host__ __device__ virtual ~Polygon_t() = default;

    // Pure virtual methods
    __host__ __device__ virtual Intersection_t intersect(const Ray_t& ray) const = 0;
    __host__ __device__ virtual void computeNormal() = 0;

    // Common methods
    __host__ __device__ const Color_t& getColor() const { return color_; }
    __host__ __device__ void setColor(const Color_t& color) { color_ = color; }

    __host__ __device__ const Vec3f_t& getNormal() {
        if (normalsDirty_) {
            computeNormal();
            normalsDirty_ = false;
        }
        return normal_;
    }

protected:
    __host__ __device__ void markNormalsDirty() { normalsDirty_ = true; }
};

// Triangle polygon implementation
class Triangle_t final : public ImplicitObject_t {
private:
    Point3f_t vertices_[3];  // Triangle vertices
    Vec3f_t normal_;

public:
__host__ __device__ Triangle_t(const Point3f_t& v0, const Point3f_t& v1, const Point3f_t& v2, Material_t* material)
    : ImplicitObject_t(material), vertices_{v0, v1, v2}{
    computeNormal();
}

    __host__ __device__ void computeNormal() {
        Vec3f_t edge1 = vertices_[1] - vertices_[0];
        Vec3f_t edge2 = vertices_[2] - vertices_[0];
        normal_ = cross(edge1, edge2).normalized();
    }

    __host__ __device__ Intersection_t intersect(const Ray_t& ray) const override {
        Intersection_t hit;
        hit.hit = false;

        // Möller–Trumbore intersection algorithm
        const Vec3f_t edge1 = vertices_[1] - vertices_[0];
        const Vec3f_t edge2 = vertices_[2] - vertices_[0];
        const Vec3f_t h = cross(ray.direction, edge2);
        const float a = dot(edge1, h);

        // Check if ray is parallel to triangle
        if (fabsf(a) < 1e-8f)
            return hit;

        const float f = 1.0f / a;
        const Vec3f_t s = ray.origin - vertices_[0];
        const float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f)
            return hit;

        const Vec3f_t q = cross(s, edge1);
        const float v = f * dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f)
            return hit;

        const float t = f * dot(edge2, q);

        if (!ray.isValidDistance(t))
            return hit;

        hit.hit = true;
        hit.distance = t;
        hit.point = ray.at(t);
        hit.material = material_;  // Add this line to set the material
        hit.setFaceNormal(ray, normal_);

        if (isEmissive_) {
            hit.emission = emissionColor_ * emissionStrength_;
        }

        return hit;
    }

    __host__ __device__ Vec3f_t getNormalAt(const Point3f_t& point) const override {
        return normal_;
    }

    __host__ __device__ const Point3f_t* getVertices() const {
        return vertices_;
    }

};

// Rectangle polygon implementation
class Rectangle_t final : public Polygon_t {
private:
    Point3f_t origin_;  // Bottom-left corner
    Vec3f_t u_;         // First edge vector
    Vec3f_t v_;         // Second edge vector

public:
    __host__ __device__ Rectangle_t(const Point3f_t& origin, const Vec3f_t& u, const Vec3f_t& v, const Color_t& color)
        : Polygon_t(color), origin_(origin), u_(u), v_(v) {
        computeNormal();
    }

    __host__ __device__ void computeNormal() override {
        normal_ = cross(u_, v_).normalized();
    }

    __host__ __device__ Intersection_t intersect(const Ray_t& ray) const override {
        Intersection_t hit;
        hit.hit = false;

        // First intersect with the plane containing the rectangle
        const float denom = dot(normal_, ray.direction);

        // Check if ray is parallel to plane
        if (fabsf(denom) < 1e-8f)
            return hit;

        const Vec3f_t p0_ro = origin_ - ray.origin;
        const float t = dot(p0_ro, normal_) / denom;

        if (!ray.isValidDistance(t))
            return hit;

        // Find hit point in plane
        const Point3f_t hit_point = ray.at(t);
        const Vec3f_t hit_vec = hit_point - origin_;

        // Project onto rectangle's coordinate system
        const float alpha = dot(hit_vec, u_) / dot(u_, u_);
        const float beta = dot(hit_vec, v_) / dot(v_, v_);

        // Check if hit point is within rectangle bounds
        if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f)
            return hit;

        hit.hit = true;
        hit.distance = t;
        hit.point = hit_point;
        hit.color = color_;
        hit.setFaceNormal(ray, normal_);

        return hit;
    }
};
