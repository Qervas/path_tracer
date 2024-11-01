#pragma once

#include <cuda_runtime.h>
#include "ImplicitObject.cuh"
#include "Polygon.cuh"
#include "Ray.cuh"
#include "Random.cuh"

// Define float max constant
constexpr float FLOAT_MAX = 3.402823466e+38f;

// Extended intersection information for scene rendering
struct SceneIntersection_t : public Intersection_t {
    const ImplicitObject_t* implicitObject;
    const Polygon_t* polygon;
    // Note: Removed material pointer as CUDA doesn't support std::shared_ptr
};

// Bounding box structure
struct BoundingBox_t {
    Point3f_t min;
    Point3f_t max;

    __host__ __device__ BoundingBox_t()
        : min(Point3f_t(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX))
        , max(Point3f_t(-FLOAT_MAX, -FLOAT_MAX, -FLOAT_MAX))
    {}
};

// GPU-friendly scene structure
struct Scene_t {
    // Device pointers to arrays of objects
    ImplicitObject_t** d_implicit_objects;
    Polygon_t** d_polygons;
    ImplicitObject_t** d_lights;
    
    // Counts of objects
    uint32_t implicit_object_count;
    uint32_t polygon_count;
    uint32_t light_count;
    
    Color_t ambient_light;
    BoundingBox_t bounds;

    // Host methods for scene setup
    __host__ Scene_t()
        : d_implicit_objects(nullptr)
        , d_polygons(nullptr)
        , d_lights(nullptr)
        , implicit_object_count(0)
        , polygon_count(0)
        , light_count(0)
        , ambient_light(0.1f, 0.1f, 0.1f)
    {}

    __host__ void allocateDeviceMemory(
        uint32_t implicit_count,
        uint32_t polygon_count,
        uint32_t light_count)
    {
        this->implicit_object_count = implicit_count;
        this->polygon_count = polygon_count;
        this->light_count = light_count;

        if (implicit_count > 0) {
            cudaMalloc(&d_implicit_objects, implicit_count * sizeof(ImplicitObject_t*));
        }
        if (polygon_count > 0) {
            cudaMalloc(&d_polygons, polygon_count * sizeof(Polygon_t*));
        }
        if (light_count > 0) {
            cudaMalloc(&d_lights, light_count * sizeof(ImplicitObject_t*));
        }
    }

    __host__ void cleanup() {
        if (d_implicit_objects) cudaFree(d_implicit_objects);
        if (d_polygons) cudaFree(d_polygons);
        if (d_lights) cudaFree(d_lights);
    }

    // Device methods for intersection and sampling
    __device__ SceneIntersection_t intersect(const Ray_t& ray) const {
        SceneIntersection_t closest_hit;
        closest_hit.hit = false;
        float closest_distance = FLOAT_MAX;

        // Check implicit objects
        for (uint32_t i = 0; i < implicit_object_count; ++i) {
            Intersection_t hit = d_implicit_objects[i]->intersect(ray);
            if (hit.hit && hit.distance < closest_distance) {
                closest_distance = hit.distance;
                closest_hit = createSceneIntersection(hit, d_implicit_objects[i], nullptr);
                
                if (d_implicit_objects[i]->isEmissive()) {
                    closest_hit.emission = d_implicit_objects[i]->getEmissionColor() * 
                                         d_implicit_objects[i]->getEmissionStrength();
                }
            }
        }

        // Check polygons
        for (uint32_t i = 0; i < polygon_count; ++i) {
            Intersection_t hit = d_polygons[i]->intersect(ray);
            if (hit.hit && hit.distance < closest_distance) {
                closest_distance = hit.distance;
                closest_hit = createSceneIntersection(hit, nullptr, d_polygons[i]);
            }
        }

        return closest_hit;
    }

    __device__ bool isVisible(const Point3f_t& from, const Point3f_t& to) const {
        Vec3f_t direction = to - from;
        float distance = direction.length();
        Ray_t shadow_ray(from, direction.normalized());
        shadow_ray.setEndpoint(to);

        // Check implicit objects
        for (uint32_t i = 0; i < implicit_object_count; ++i) {
            Intersection_t hit = d_implicit_objects[i]->intersect(shadow_ray);
            if (hit.hit && hit.distance < distance) return false;
        }

        // Check polygons
        for (uint32_t i = 0; i < polygon_count; ++i) {
            Intersection_t hit = d_polygons[i]->intersect(shadow_ray);
            if (hit.hit && hit.distance < distance) return false;
        }

        return true;
    }

    __device__ Point3f_t sampleLight(curandState* rand_state) const {
        if (light_count == 0) return Point3f_t();
        
        uint32_t light_idx = curand(rand_state) % light_count;
        const ImplicitObject_t* light = d_lights[light_idx];
        
        // Assuming lights are spheres for now
        float u = Random_t::get(rand_state);
        float v = Random_t::get(rand_state);
        
        // Sample point on sphere surface
        float theta = 2.0f * M_PI * u;
        float phi = acosf(2.0f * v - 1.0f);
        float r = light->getRadius();
        
        return light->getCenter() + Vec3f_t(
            r * sinf(phi) * cosf(theta),
            r * sinf(phi) * sinf(theta),
            r * cosf(phi)
        );
    }

private:
    __device__ static SceneIntersection_t createSceneIntersection(
        const Intersection_t& hit,
        const ImplicitObject_t* implicit_obj,
        const Polygon_t* polygon_obj)
    {
        SceneIntersection_t scene_hit;
        scene_hit.hit = hit.hit;
        scene_hit.point = hit.point;
        scene_hit.normal = hit.normal;
        scene_hit.distance = hit.distance;
        scene_hit.color = hit.color;
        scene_hit.frontFace = hit.frontFace;
        scene_hit.emission = hit.emission;
        scene_hit.implicitObject = implicit_obj;
        scene_hit.polygon = polygon_obj;
        return scene_hit;
    }
};

// Helper class for managing scene data transfer
class SceneManager {
private:
    Scene_t h_scene;  // Host-side scene data
    Scene_t* d_scene; // Device-side scene data

public:
    __host__ SceneManager() : d_scene(nullptr) {}

    __host__ ~SceneManager() {
        if (d_scene) {
            h_scene.cleanup();
            cudaFree(d_scene);
        }
    }

    __host__ void initializeScene(
        const std::vector<ImplicitObject_t*>& implicit_objects,
        const std::vector<Polygon_t*>& polygons,
        const std::vector<ImplicitObject_t*>& lights)
    {
        // Allocate host-side arrays
        h_scene.allocateDeviceMemory(
            implicit_objects.size(),
            polygons.size(),
            lights.size()
        );

        // Copy objects to device
        if (!implicit_objects.empty()) {
            cudaMemcpy(h_scene.d_implicit_objects, implicit_objects.data(),
                      implicit_objects.size() * sizeof(ImplicitObject_t*),
                      cudaMemcpyHostToDevice);
        }
        if (!polygons.empty()) {
            cudaMemcpy(h_scene.d_polygons, polygons.data(),
                      polygons.size() * sizeof(Polygon_t*),
                      cudaMemcpyHostToDevice);
        }
        if (!lights.empty()) {
            cudaMemcpy(h_scene.d_lights, lights.data(),
                      lights.size() * sizeof(ImplicitObject_t*),
                      cudaMemcpyHostToDevice);
        }

        // Copy scene structure to device
        cudaMalloc(&d_scene, sizeof(Scene_t));
        cudaMemcpy(d_scene, &h_scene, sizeof(Scene_t), cudaMemcpyHostToDevice);
    }

    __host__ Scene_t* getDeviceScene() const { return d_scene; }
}; 