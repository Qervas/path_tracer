#pragma once

#include "ImplicitObject.h"
#include "Polygon.h"
#include "Ray.h"
#include "Random.h"
#include <vector>
#include <memory>
#include <optional>

// Forward declarations
struct SceneIntersection;
class Material;

// Extended intersection information for scene rendering
struct SceneIntersection : public Intersection {
    const ImplicitObject* implicitObject{nullptr};
    const Polygon* polygon{nullptr};
    std::shared_ptr<Material> material;
};

class Scene {
private:
    std::vector<std::shared_ptr<ImplicitObject>> implicit_objects_;
    std::vector<std::shared_ptr<Polygon>> polygons_;
    std::vector<std::shared_ptr<ImplicitObject>> lights_;  // Emissive objects
    ColorDBL ambient_light_{0.1, 0.1, 0.1};
    
    // Acceleration structure (could be expanded later)
    struct BoundingBox {
        Point3d min, max;
    } scene_bounds_;

public:
    Scene() = default;

    // Object management
    void addImplicitObject(std::shared_ptr<ImplicitObject> object) {
        implicit_objects_.push_back(object);
        if (object->isEmissive()) {
            lights_.push_back(object);
        }
        updateBounds();
    }

    void addPolygon(std::shared_ptr<Polygon> polygon) {
        polygons_.push_back(polygon);
        updateBounds();
    }

    // Find the closest intersection in the scene
    [[nodiscard]] std::optional<SceneIntersection> intersect(const Ray& ray) const {
        std::optional<SceneIntersection> closest_hit;
        double closest_distance = std::numeric_limits<double>::infinity();

        // Check implicit objects
        for (const auto& object : implicit_objects_) {
            if (auto hit = object->intersect(ray)) {
                if (hit->distance < closest_distance) {
                    closest_distance = hit->distance;
                    closest_hit = createSceneIntersection(*hit, object.get(), nullptr);
                    
                    // Set emission for emissive objects
                    if (object->isEmissive()) {
                        closest_hit->emission = object->getEmissionColor() * object->getEmissionStrength();
                    }
                }
            }
        }

        return closest_hit;
    }

    // Check if a point is visible from another point (shadow ray)
    [[nodiscard]] bool isVisible(const Point3d& from, const Point3d& to) const {
        Vec3d direction = to - from;
        double distance = direction.length();
        Ray shadow_ray(from, direction.normalized());
        shadow_ray.setEndpoint(to);

        for (const auto& object : implicit_objects_) {
            if (auto hit = object->intersect(shadow_ray)) {
                if (hit->distance < distance) return false;
            }
        }

        for (const auto& polygon : polygons_) {
            if (auto hit = polygon->intersect(shadow_ray)) {
                if (hit->distance < distance) return false;
            }
        }

        return true;
    }

    // Light sampling methods
    [[nodiscard]] std::vector<Point3d> sampleLights(uint32_t samples_per_light) const {
        std::vector<Point3d> light_samples;
        light_samples.reserve(lights_.size() * samples_per_light);

        for (const auto& light : lights_) {
            if (auto sphere = dynamic_cast<const Sphere*>(light.get())) {
                for (uint32_t i = 0; i < samples_per_light; ++i) {
                    double u = Random::get();
                    double v = Random::get();
                    light_samples.push_back(sphere->getRandomPoint(u, v));
                }
            }
        }

        return light_samples;
    }

    // Scene bounds management
    void updateBounds() {
        // Reset bounds
        scene_bounds_.min = Point3d(std::numeric_limits<double>::max());
        scene_bounds_.max = Point3d(std::numeric_limits<double>::lowest());

        // Update with implicit objects
        for (const auto& object : implicit_objects_) {
            if (auto sphere = dynamic_cast<const Sphere*>(object.get())) {
                updateBoundsWithSphere(*sphere);
            }
        }

        // Update with polygons (assuming axis-aligned bounding box)
        for (const auto& polygon : polygons_) {
            if (auto triangle = dynamic_cast<const Triangle*>(polygon.get())) {
                updateBoundsWithTriangle(*triangle);
            }
        }
    }

    // Getters
    [[nodiscard]] const std::vector<std::shared_ptr<ImplicitObject>>& getLights() const {
        return lights_;
    }

    [[nodiscard]] const ColorDBL& getAmbientLight() const { return ambient_light_; }
    void setAmbientLight(const ColorDBL& ambient) { ambient_light_ = ambient; }

private:
    static SceneIntersection createSceneIntersection(
        const Intersection& hit,
        const ImplicitObject* implicit_obj,
        const Polygon* polygon_obj
    ) {
        SceneIntersection scene_hit;
        scene_hit.point = hit.point;
        scene_hit.normal = hit.normal;
        scene_hit.distance = hit.distance;
        scene_hit.color = hit.color;
        scene_hit.frontFace = hit.frontFace;
        scene_hit.implicitObject = implicit_obj;
        scene_hit.polygon = polygon_obj;
        return scene_hit;
    }

    void updateBoundsWithSphere(const Sphere& sphere) {
        Vec3d radius_vec(sphere.getRadius());
        Point3d center = sphere.getCenter();
        scene_bounds_.min = min(scene_bounds_.min, center - radius_vec);
        scene_bounds_.max = max(scene_bounds_.max, center + radius_vec);
    }

    void updateBoundsWithTriangle(const Triangle& triangle) {
        const Point3d* vertices = triangle.getVertices();
        
        // Update bounds with each vertex
        for (int i = 0; i < 3; ++i) {
            scene_bounds_.min = min(scene_bounds_.min, vertices[i]);
            scene_bounds_.max = max(scene_bounds_.max, vertices[i]);
        }
    }
}; 