#pragma once

#include "Vec3.cuh"
#include "Color.cuh"
#include "Ray.cuh"
#include "ForwardDeclarations.cuh"

struct Intersection_t {
    Point3f_t point;      // Intersection point
    Vec3f_t normal;       // Surface normal at intersection
    float distance;       // Distance from ray origin
    Color_t color;          // Surface color at intersection
    bool frontFace;       // Whether the intersection is on the front face
    Color_t emission;       // Emission color (for emissive objects)
    bool hit;             // Whether there was an intersection
    Material_t* material;  // Material at intersection

    __host__ __device__ Intersection_t()
        : point()
        , normal()
        , distance(1e6f)
        , color()
        , frontFace(false)
        , emission()
        , hit(false)
        , material(nullptr)
    {}

    __host__ __device__ void setFaceNormal(const Ray_t& ray, const Vec3f_t& outwardNormal) {
        frontFace = dot(ray.direction, outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }

    __host__ __device__ static Intersection_t miss() {
        return Intersection_t();
    }
}; 