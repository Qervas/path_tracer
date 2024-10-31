#pragma once

#include "Vec3.h"
#include "Color.h"
#include "Ray.h"

struct Intersection {
    Point3d point;        // Intersection point
    Vec3d normal;         // Surface normal at intersection
    double distance;      // Distance from ray origin
    ColorDBL color;       // Surface color at intersection
    bool frontFace;       // Whether the intersection is on the front face
    ColorDBL emission;    // Emission color (for emissive objects)

    // Set the normal vector, ensuring correct orientation
    void setFaceNormal(const Ray& ray, const Vec3d& outwardNormal) {
        frontFace = dot(ray.direction(), outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
}; 