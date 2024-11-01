#pragma once

#include "Vec3.cuh"
#include "Color.cuh"
#include "Ray.cuh"

struct Intersection_t {
    Point3f_t point;      // Intersection point
    Vec3f_t normal;       // Surface normal at intersection
    float distance;       // Distance from ray origin
    Color_t color;          // Surface color at intersection
    bool frontFace;       // Whether the intersection is on the front face
    Color_t emission;       // Emission color (for emissive objects)
    bool hit;             // Whether there was an intersection

    // Constructor
    __host__ __device__ Intersection_t()
        : point()
        , normal()
        , distance(1e6f)
        , color()
        , frontFace(false)
        , emission()
        , hit(false)
    {}

    // Set the normal vector, ensuring correct orientation
    __host__ __device__ void setFaceNormal(const Ray_t& ray, const Vec3f_t& outwardNormal) {
        frontFace = dot(ray.direction, outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }

    // Utility function to create a miss intersection
    __host__ __device__ static Intersection_t miss() {
        return Intersection_t();
    }

    // Utility function to check if this is closer than another intersection
    __host__ __device__ bool isCloser(const Intersection_t& other) const {
        return distance < other.distance;
    }
}; 