#pragma once

#include "Polygon.cuh"
#include "Vec3.cuh"
#include "Ray.cuh"
#include "Triangle.cuh"

class Tetrahedron_t final : public Polygon_t {
private:
    Point3f_t vertices_[4];     // Four vertices of the tetrahedron
    Vec3f_t face_normals_[4];   // Normals for each face
    Triangle_t faces_[4];       // Cached triangle faces

    // Define face indices as constant array
    static constexpr uint8_t FACE_INDICES[4][3] = {
        {0, 2, 1},  // Bottom face
        {0, 1, 3},  // Front face
        {1, 2, 3},  // Right face
        {2, 0, 3}   // Left face
    };

public:
    __host__ __device__ Tetrahedron_t(const Point3f_t vertices[4], const Color_t& color)
        : Polygon_t(color)
    {
        // Copy vertices
        for (int i = 0; i < 4; ++i) {
            vertices_[i] = vertices[i];
        }

        // Initialize faces
        initializeFaces();
        computeNormal();
    }

    // Create regular tetrahedron centered at origin with given radius
    __host__ static Tetrahedron_t createRegular(float radius, const Color_t& color) {
        const float a = radius * 2.0f / sqrtf(6.0f);
        const float h = radius * 3.0f / sqrtf(6.0f);
        
        Point3f_t vertices[4] = {
            Point3f_t(0, -radius, -h/3),           // Bottom center
            Point3f_t(-a/2, radius/3, -h/3),       // Back left
            Point3f_t(a/2, radius/3, -h/3),        // Back right
            Point3f_t(0, 0, 2*h/3)                 // Top
        };

        return Tetrahedron_t(vertices, color);
    }

private:
    __host__ __device__ void initializeFaces() {
        // Initialize the four triangular faces
        for (int i = 0; i < 4; ++i) {
            faces_[i] = Triangle_t(
                vertices_[FACE_INDICES[i][0]],
                vertices_[FACE_INDICES[i][1]],
                vertices_[FACE_INDICES[i][2]],
                color_
            );
        }
    }

    __host__ __device__ void computeNormal() override {
        // Compute normals for each face
        for (int i = 0; i < 4; ++i) {
            Vec3f_t edge1 = vertices_[FACE_INDICES[i][1]] - vertices_[FACE_INDICES[i][0]];
            Vec3f_t edge2 = vertices_[FACE_INDICES[i][2]] - vertices_[FACE_INDICES[i][0]];
            face_normals_[i] = cross(edge1, edge2).normalized();
        }
    }

    __host__ __device__ bool containsPoint(const Point3f_t& point) const {
        for (int i = 0; i < 4; ++i) {
            if (dot(face_normals_[i], point - vertices_[FACE_INDICES[i][0]]) > 0) {
                return false;
            }
        }
        return true;
    }

public:
    __host__ __device__ Intersection_t intersect(const Ray_t& ray) const override {
        Intersection_t closest_hit;
        closest_hit.hit = false;
        float min_distance = 1e6f;

        for (int i = 0; i < 4; ++i) {
            Intersection_t hit = faces_[i].intersect(ray);
            if (hit.hit && hit.distance < min_distance) {
                min_distance = hit.distance;
                closest_hit = hit;
            }
        }

        return closest_hit;
    }

    __host__ __device__ Vec3f_t getFaceNormal(const Point3f_t& point) const {
        for (int i = 0; i < 4; ++i) {
            Ray_t test_ray(point + face_normals_[i] * 0.001f, -face_normals_[i]);
            Intersection_t hit = faces_[i].intersect(test_ray);
            if (hit.hit && hit.distance < 0.002f) {
                return face_normals_[i];
            }
        }
        return Vec3f_t(0, 1, 0); // Fallback normal
    }

    __host__ __device__ float getVolume() const {
        Vec3f_t v01 = vertices_[1] - vertices_[0];
        Vec3f_t v02 = vertices_[2] - vertices_[0];
        Vec3f_t v03 = vertices_[3] - vertices_[0];
        return fabsf(dot(cross(v01, v02), v03)) / 6.0f;
    }

    __host__ __device__ float getSurfaceArea() const {
        float total_area = 0.0f;
        for (int i = 0; i < 4; ++i) {
            Vec3f_t edge1 = vertices_[1] - vertices_[0];
            Vec3f_t edge2 = vertices_[2] - vertices_[0];
            total_area += cross(edge1, edge2).length() / 2.0f;
        }
        return total_area;
    }

    __host__ __device__ Point3f_t getCentroid() const {
        return (vertices_[0] + vertices_[1] + vertices_[2] + vertices_[3]) * 0.25f;
    }

    // Transform methods
    __host__ __device__ void translate(const Vec3f_t& offset) {
        for (int i = 0; i < 4; ++i) {
            vertices_[i] += offset;
        }
        initializeFaces();
        computeNormal();
    }

    __host__ __device__ void scale(float factor) {
        Point3f_t centroid = getCentroid();
        for (int i = 0; i < 4; ++i) {
            vertices_[i] = centroid + (vertices_[i] - centroid) * factor;
        }
        initializeFaces();
        computeNormal();
    }

    __host__ __device__ void rotate(const Vec3f_t& axis, float angle) {
        float half_angle = angle * 0.5f;
        float sin_half = sinf(half_angle);
        
        Vec3f_t normalized_axis = axis.normalized();
        Vec3f_t q_vector = normalized_axis * sin_half;
        float q_scalar = cosf(half_angle);
        
        Point3f_t centroid = getCentroid();
        for (int i = 0; i < 4; ++i) {
            Vec3f_t v = vertices_[i] - centroid;
            
            // Apply quaternion rotation: q * v * q^-1
            Vec3f_t t = cross(q_vector, v) * 2.0f;
            vertices_[i] = centroid + v + q_scalar * t + cross(q_vector, t);
        }
        
        initializeFaces();
        computeNormal();
    }

    // Getter for vertices
    __host__ __device__ const Point3f_t* getVertices() const {
        return vertices_;
    }
}; 