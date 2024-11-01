#pragma once

#include "Vec3.cuh"
#include "Ray.cuh"
#include "Color.cuh"
#include "Intersection.cuh"

class Triangle_t {
private:
    Point3f_t vertices_[3];  // Triangle vertices
    Vec3f_t normal_;         // Cached normal vector
    Color_t color_;          // Triangle color

public:
    __host__ __device__ Triangle_t() = default;

    __host__ __device__ Triangle_t(const Point3f_t& v0, const Point3f_t& v1, const Point3f_t& v2, const Color_t& color)
        : vertices_{v0, v1, v2}
        , color_(color)
    {
        computeNormal();
    }

    __host__ __device__ void computeNormal() {
        Vec3f_t edge1 = vertices_[1] - vertices_[0];
        Vec3f_t edge2 = vertices_[2] - vertices_[0];
        normal_ = cross(edge1, edge2).normalized();
    }

    __host__ __device__ Intersection_t intersect(const Ray_t& ray) const {
        Intersection_t hit;
        hit.hit = false;

        // Möller–Trumbore intersection algorithm
        const Vec3f_t edge1 = vertices_[1] - vertices_[0];
        const Vec3f_t edge2 = vertices_[2] - vertices_[0];
        const Vec3f_t h = cross(ray.direction, edge2);
        const float a = dot(edge1, h);

        // Check if ray is parallel to triangle
        if (fabsf(a) < 1e-8f) {
            return hit;
        }

        const float f = 1.0f / a;
        const Vec3f_t s = ray.origin - vertices_[0];
        const float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f) {
            return hit;
        }

        const Vec3f_t q = cross(s, edge1);
        const float v = f * dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f) {
            return hit;
        }

        const float t = f * dot(edge2, q);

        if (!ray.isValidDistance(t)) {
            return hit;
        }

        hit.hit = true;
        hit.distance = t;
        hit.point = ray.at(t);
        hit.normal = normal_;
        hit.color = color_;
        hit.setFaceNormal(ray, normal_);

        return hit;
    }

    // Getters
    __host__ __device__ const Point3f_t* getVertices() const { 
        return vertices_; 
    }

    __host__ __device__ const Vec3f_t& getNormal() const { 
        return normal_; 
    }

    __host__ __device__ const Color_t& getColor() const { 
        return color_; 
    }

    // Utility methods
    __host__ __device__ float getArea() const {
        Vec3f_t edge1 = vertices_[1] - vertices_[0];
        Vec3f_t edge2 = vertices_[2] - vertices_[0];
        return cross(edge1, edge2).length() * 0.5f;
    }

    __host__ __device__ Point3f_t getCentroid() const {
        return (vertices_[0] + vertices_[1] + vertices_[2]) * (1.0f / 3.0f);
    }

    // Transform methods
    __host__ __device__ void translate(const Vec3f_t& offset) {
        vertices_[0] += offset;
        vertices_[1] += offset;
        vertices_[2] += offset;
    }

    __host__ __device__ void scale(float factor, const Point3f_t& center) {
        vertices_[0] = center + (vertices_[0] - center) * factor;
        vertices_[1] = center + (vertices_[1] - center) * factor;
        vertices_[2] = center + (vertices_[2] - center) * factor;
        computeNormal();
    }

    __host__ __device__ bool containsPoint(const Point3f_t& point) const {
        // Compute barycentric coordinates
        Vec3f_t v0 = vertices_[1] - vertices_[0];
        Vec3f_t v1 = vertices_[2] - vertices_[0];
        Vec3f_t v2 = point - vertices_[0];

        float d00 = dot(v0, v0);
        float d01 = dot(v0, v1);
        float d11 = dot(v1, v1);
        float d20 = dot(v2, v0);
        float d21 = dot(v2, v1);

        float denom = d00 * d11 - d01 * d01;
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;

        return v >= 0.0f && w >= 0.0f && (v + w) <= 1.0f;
    }
}; 