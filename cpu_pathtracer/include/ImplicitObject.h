#pragma once

#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "Intersection.h"
#include <optional>
#include <memory>

// Forward declaration
class Ray;

// Base class for all implicit objects
class ImplicitObject {
protected:
    ColorDBL color_;
    bool isEmissive_{false};
    ColorDBL emissionColor_;
    double emissionStrength_{0.0};

public:
    explicit ImplicitObject(const ColorDBL& color) noexcept : color_(color) {}
    virtual ~ImplicitObject() = default;

    // Pure virtual methods
    [[nodiscard]] virtual std::optional<Intersection> intersect(const Ray& ray) const = 0;
    [[nodiscard]] virtual Vec3d getNormalAt(const Point3d& point) const = 0;

    // Common methods
    void makeEmissive(const ColorDBL& emission_color, double strength) noexcept {
        isEmissive_ = true;
        emissionColor_ = emission_color;
        emissionStrength_ = strength;
    }

    [[nodiscard]] bool isEmissive() const noexcept { return isEmissive_; }
    [[nodiscard]] const ColorDBL& getEmissionColor() const noexcept { return emissionColor_; }
    [[nodiscard]] double getEmissionStrength() const noexcept { return emissionStrength_; }
    [[nodiscard]] const ColorDBL& getColor() const noexcept { return color_; }
};

// Sphere implementation
class Sphere final : public ImplicitObject {
private:
    Point3d center_;
    double radius_;
    double radiusSquared_;  // Cached for performance

public:
    Sphere(const Point3d& center, double radius, const ColorDBL& color)
        : ImplicitObject(color)
        , center_(center)
        , radius_(radius)
        , radiusSquared_(radius * radius)
    {}

    [[nodiscard]] std::optional<Intersection> intersect(const Ray& ray) const override {
        // Optimized ray-sphere intersection
        const Vec3d oc = ray.origin() - center_;
        const double a = 1.0;  // Optimized since ray direction is normalized
        const double half_b = dot(oc, ray.direction());
        const double c = dot(oc, oc) - radiusSquared_;
        const double discriminant = half_b * half_b - a * c;

        if (discriminant < 0) {
            return std::nullopt;  // No intersection
        }

        // Find the nearest valid intersection
        const double sqrtd = std::sqrt(discriminant);
        double root = (-half_b - sqrtd) / a;

        // Check if the nearest intersection is valid
        if (!ray.isValidDistance(root)) {
            root = (-half_b + sqrtd) / a;
            if (!ray.isValidDistance(root)) {
                return std::nullopt;
            }
        }

        // Create intersection data
        Intersection hit;
        hit.distance = root;
        hit.point = ray.at(root);
        hit.normal = getNormalAt(hit.point);
        hit.color = color_;
        hit.setFaceNormal(ray, hit.normal);

        // Add emission information if the sphere is emissive
        if (isEmissive_) {
            hit.emission = emissionColor_ * emissionStrength_;
        }

        return hit;
    }

    [[nodiscard]] Vec3d getNormalAt(const Point3d& point) const override {
        return (point - center_) / radius_;
    }

    // Sphere-specific methods
    [[nodiscard]] const Point3d& getCenter() const noexcept { return center_; }
    [[nodiscard]] double getRadius() const noexcept { return radius_; }

    // Check if a point lies inside the sphere
    [[nodiscard]] bool contains(const Point3d& point) const noexcept {
        return (point - center_).length_squared() <= radiusSquared_;
    }

    // Get a random point on the sphere surface (useful for light sampling)
    [[nodiscard]] Point3d getRandomPoint(double u, double v) const noexcept {
        // Convert uniform random variables to spherical coordinates
        const double theta = 2.0 * M_PI * u;
        const double phi = std::acos(2.0 * v - 1.0);
        
        // Convert to Cartesian coordinates
        const double sin_phi = std::sin(phi);
        const Vec3d direction(
            std::cos(theta) * sin_phi,
            std::sin(theta) * sin_phi,
            std::cos(phi)
        );
        
        return center_ + direction * radius_;
    }

    // Get the surface area of the sphere
    [[nodiscard]] double getSurfaceArea() const noexcept {
        return 4.0 * M_PI * radiusSquared_;
    }
};

// Example of another implicit object: Infinite Plane
class Plane final : public ImplicitObject {
private:
    Point3d point_;   // A point on the plane
    Vec3d normal_;    // Plane normal vector

public:
    Plane(const Point3d& point, const Vec3d& normal, const ColorDBL& color)
        : ImplicitObject(color)
        , point_(point)
        , normal_(normal.normalized())
    {}

    [[nodiscard]] std::optional<Intersection> intersect(const Ray& ray) const override {
        const double denom = dot(normal_, ray.direction());
        
        // Check if ray is parallel to plane
        if (std::abs(denom) < 1e-8) {
            return std::nullopt;
        }

        const double t = dot(point_ - ray.origin(), normal_) / denom;
        
        if (!ray.isValidDistance(t)) {
            return std::nullopt;
        }

        Intersection hit;
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

    [[nodiscard]] Vec3d getNormalAt(const Point3d&) const override {
        return normal_;
    }
}; 