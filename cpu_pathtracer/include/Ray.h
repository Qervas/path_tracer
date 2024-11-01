#pragma once

#include "Vec3.h"
#include "Color.h"
#include <memory>
#include <optional>
#include <vector>

// Forward declarations
class Polygon;
struct Intersection;  

class Ray : public std::enable_shared_from_this<Ray> {
public:
    // Ray types for optimization and debugging
    enum class Type {
        PRIMARY,    // Camera rays
        REFLECTED,  // Specular reflection
        REFRACTED,  // Transmission/refraction
        SHADOW,     // Shadow testing
        DIFFUSE     // Diffuse bounce
    };

private:
    Point3d origin_;                    // Ray origin
    Vec3d direction_;                   // Normalized direction vector
    std::optional<Point3d> endpoint_;   // Optional endpoint (for finite rays)
    
    ColorDBL color_;                    // Current ray color
    double importance_;                 // Ray importance for Russian Roulette
    
    Type type_;                         // Ray type for optimization
    uint32_t depth_;                    // Bounce depth
    
    // Surface tracking
    const Polygon* surface_;            // Surface where ray originated
    
    // Path tracking
    std::shared_ptr<Ray> prev_ray_;     // Previous ray in path
    mutable std::shared_ptr<Ray> next_ray_;  // Make mutable to allow modification in const methods
    
    // Optional parameters
    double min_t_{1e-8};               // Minimum intersection distance
    double max_t_{1e30};               // Maximum intersection distance

public:
    // Constructor for new primary rays
    Ray(const Point3d& origin, 
        const Vec3d& direction,
        Type type = Type::PRIMARY) noexcept
        : origin_(origin)
        , direction_(direction.normalized())
        , color_(ColorDBL(1.0, 1.0, 1.0))
        , importance_(1.0)
        , type_(type)
        , depth_(0)
        , surface_(nullptr)
    {}

    // Constructor for secondary rays (bounces)
    Ray(const Point3d& origin,
        const Vec3d& direction,
        const ColorDBL& color,
        double importance,
        Type type,
        uint32_t depth,
        const Polygon* surface,
        std::shared_ptr<Ray> prev_ray) noexcept
        : origin_(origin)
        , direction_(direction.normalized())
        , color_(color)
        , importance_(importance)
        , type_(type)
        , depth_(depth)
        , surface_(surface)
        , prev_ray_(prev_ray)
    {}

    // Factory methods for creating new rays in the path
    [[nodiscard]] std::shared_ptr<Ray> createReflectedRay(
        const Point3d& origin,
        const Vec3d& direction,
        const ColorDBL& attenuation,
        const Polygon* surface) const {
        auto new_ray = std::make_shared<Ray>(
            origin,
            direction,
            color_ * attenuation,
            importance_ * 0.8,
            Type::REFLECTED,
            depth_ + 1,
            surface,
            std::const_pointer_cast<Ray>(shared_from_this())
        );
        next_ray_ = new_ray;
        return new_ray;
    }

    [[nodiscard]] std::shared_ptr<Ray> createRefractedRay(
        const Point3d& origin,
        const Vec3d& direction,
        const ColorDBL& attenuation,
        const Polygon* surface) const {
        auto new_ray = std::make_shared<Ray>(
            origin,
            direction,
            color_ * attenuation,
            importance_ * 0.8,
            Type::REFRACTED,
            depth_ + 1,
            surface,
            std::const_pointer_cast<Ray>(shared_from_this())
        );
        next_ray_ = new_ray;
        return new_ray;
    }

    [[nodiscard]] std::shared_ptr<Ray> createDiffuseRay(
        const Point3d& origin,
        const Vec3d& direction,
        const ColorDBL& attenuation,
        const Polygon* surface) const {
        auto new_ray = std::make_shared<Ray>(
            origin,
            direction,
            color_ * attenuation,
            importance_ * 0.5,
            Type::DIFFUSE,
            depth_ + 1,
            surface,
            std::const_pointer_cast<Ray>(shared_from_this())
        );
        next_ray_ = new_ray;
        return new_ray;
    }

    [[nodiscard]] std::shared_ptr<Ray> createShadowRay(
        const Point3d& origin,
        const Point3d& target,
        const Polygon* surface) const {
        Vec3d direction = (target - origin).normalized();
        
        auto shadow_ray = std::make_shared<Ray>(
            origin,
            direction,
            ColorDBL(1.0, 1.0, 1.0),
            1.0,
            Type::SHADOW,
            depth_,
            surface,
            std::const_pointer_cast<Ray>(shared_from_this())
        );
        shadow_ray->setEndpoint(target);
        return shadow_ray;
    }

    // Accessors
    [[nodiscard]] const Point3d& origin() const noexcept { return origin_; }
    [[nodiscard]] const Vec3d& direction() const noexcept { return direction_; }
    [[nodiscard]] const ColorDBL& color() const noexcept { return color_; }
    [[nodiscard]] double importance() const noexcept { return importance_; }
    [[nodiscard]] Type type() const noexcept { return type_; }
    [[nodiscard]] uint32_t depth() const noexcept { return depth_; }
    [[nodiscard]] const Polygon* surface() const noexcept { return surface_; }
    
    // Path tracking
    [[nodiscard]] const std::shared_ptr<Ray>& prevRay() const noexcept { return prev_ray_; }
    [[nodiscard]] const std::shared_ptr<Ray>& nextRay() const noexcept { return next_ray_; }

    // Utility methods
    [[nodiscard]] Point3d at(double t) const noexcept {
        return origin_ + direction_ * t;
    }

    void setEndpoint(const Point3d& endpoint) noexcept {
        endpoint_ = endpoint;
        max_t_ = (endpoint - origin_).length();
    }

    [[nodiscard]] bool hasEndpoint() const noexcept {
        return endpoint_.has_value();
    }

    [[nodiscard]] const Point3d& getEndpoint() const {
        if (!endpoint_) {
            throw std::runtime_error("No endpoint set");
        }
        return *endpoint_;
    }

    // Distance validity checking
    [[nodiscard]] bool isValidDistance(double t) const noexcept {
        return t >= min_t_ && t <= max_t_;
    }

    // Ray path utilities
    [[nodiscard]] std::vector<std::shared_ptr<Ray>> getFullPath() const {
        std::vector<std::shared_ptr<Ray>> path;
        
        // Track backwards to find start
        const Ray* current = this;
        while (current->prev_ray_) {
            current = current->prev_ray_.get();
        }
        
        // Now go forward and collect all rays
        while (current) {
            path.push_back(std::shared_ptr<Ray>(const_cast<Ray*>(current)));
            current = current->next_ray_.get();
        }
        
        return path;
    }
}; 