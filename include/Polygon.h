#pragma once

#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "Intersection.h"
#include <optional>
#include <memory>

// Abstract base class for all polygons
class Polygon {
protected:
    ColorDBL color_;
    Vec3d normal_;
    bool normalsDirty_{true};  // Flag for lazy normal computation

public:
    explicit Polygon(const ColorDBL& color) noexcept : color_(color) {}
    virtual ~Polygon() = default;

    // Pure virtual methods
    [[nodiscard]] virtual std::optional<Intersection> intersect(const Ray& ray) const = 0;
    virtual void computeNormal() = 0;

    // Common methods
    [[nodiscard]] const ColorDBL& getColor() const noexcept { return color_; }
    void setColor(const ColorDBL& color) noexcept { color_ = color; }

    [[nodiscard]] const Vec3d& getNormal() {
        if (normalsDirty_) {
            computeNormal();
            normalsDirty_ = false;
        }
        return normal_;
    }

protected:
    void markNormalsDirty() noexcept { normalsDirty_ = true; }
};

// Triangle polygon implementation
class Triangle final : public Polygon {
private:
    Point3d vertices_[3];  // Triangle vertices

public:
    Triangle(const Point3d& v0, const Point3d& v1, const Point3d& v2, const ColorDBL& color)
        : Polygon(color), vertices_{v0, v1, v2} {
        computeNormal();
    }

    void computeNormal() override {
        Vec3d edge1 = vertices_[1] - vertices_[0];
        Vec3d edge2 = vertices_[2] - vertices_[0];
        normal_ = cross(edge1, edge2).normalized();
    }

    [[nodiscard]] std::optional<Intersection> intersect(const Ray& ray) const override {
        // Möller–Trumbore intersection algorithm
        const Vec3d edge1 = vertices_[1] - vertices_[0];
        const Vec3d edge2 = vertices_[2] - vertices_[0];
        const Vec3d h = cross(ray.direction(), edge2);
        const double a = dot(edge1, h);

        // Check if ray is parallel to triangle
        if (std::abs(a) < 1e-8)
            return std::nullopt;

        const double f = 1.0 / a;
        const Vec3d s = ray.origin() - vertices_[0];
        const double u = f * dot(s, h);

        if (u < 0.0 || u > 1.0)
            return std::nullopt;

        const Vec3d q = cross(s, edge1);
        const double v = f * dot(ray.direction(), q);

        if (v < 0.0 || u + v > 1.0)
            return std::nullopt;

        const double t = f * dot(edge2, q);

        if (t < 1e-8)
            return std::nullopt;

        Intersection hit;
        hit.distance = t;
        hit.point = ray.at(t);
        hit.color = color_;
        hit.setFaceNormal(ray, normal_);

        return hit;
    }

    // Get triangle vertices
    [[nodiscard]] const Point3d* getVertices() const noexcept { 
        return vertices_; 
    }
};

// Rectangle polygon implementation
class Rectangle final : public Polygon {
private:
    Point3d origin_;     // Bottom-left corner
    Vec3d u_;           // First edge vector
    Vec3d v_;           // Second edge vector

public:
    Rectangle(const Point3d& origin, const Vec3d& u, const Vec3d& v, const ColorDBL& color)
        : Polygon(color), origin_(origin), u_(u), v_(v) {
        computeNormal();
    }

    void computeNormal() override {
        normal_ = cross(u_, v_).normalized();
    }

    [[nodiscard]] std::optional<Intersection> intersect(const Ray& ray) const override {
        // First intersect with the plane containing the rectangle
        const double denom = dot(normal_, ray.direction());
        
        // Check if ray is parallel to plane
        if (std::abs(denom) < 1e-8)
            return std::nullopt;

        const Vec3d p0_ro = origin_ - ray.origin();
        const double t = dot(p0_ro, normal_) / denom;

        if (t < 1e-8)
            return std::nullopt;

        // Find hit point in plane
        const Point3d hit_point = ray.at(t);
        const Vec3d hit_vec = hit_point - origin_;

        // Project onto rectangle's coordinate system
        const double alpha = dot(hit_vec, u_) / dot(u_, u_);
        const double beta = dot(hit_vec, v_) / dot(v_, v_);

        // Check if hit point is within rectangle bounds
        if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0)
            return std::nullopt;

        Intersection hit;
        hit.distance = t;
        hit.point = hit_point;
        hit.color = color_;
        hit.setFaceNormal(ray, normal_);

        return hit;
    }
}; 