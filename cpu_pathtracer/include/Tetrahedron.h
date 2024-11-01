#pragma once

#include "Polygon.h"
#include "Vec3.h"
#include "Ray.h"
#include <array>
#include <algorithm>

class Tetrahedron final : public Polygon {
private:
    std::array<Point3d, 4> vertices_;  // Four vertices of the tetrahedron
    std::array<Vec3d, 4> face_normals_; // Normals for each face
    std::array<Triangle, 4> faces_;    // Cached triangle faces

    // Define face indices directly as a static array
    static constexpr uint8_t FACE_INDICES[4][3] = {
        {0, 2, 1},  // Bottom face
        {0, 1, 3},  // Front face
        {1, 2, 3},  // Right face
        {2, 0, 3}   // Left face
    };

public:
    Tetrahedron(const std::array<Point3d, 4>& vertices, const ColorDBL& color)
        : Polygon(color)
        , vertices_(vertices)
        , faces_{{
            Triangle(vertices[FACE_INDICES[0][0]], vertices[FACE_INDICES[0][1]], vertices[FACE_INDICES[0][2]], color),
            Triangle(vertices[FACE_INDICES[1][0]], vertices[FACE_INDICES[1][1]], vertices[FACE_INDICES[1][2]], color),
            Triangle(vertices[FACE_INDICES[2][0]], vertices[FACE_INDICES[2][1]], vertices[FACE_INDICES[2][2]], color),
            Triangle(vertices[FACE_INDICES[3][0]], vertices[FACE_INDICES[3][1]], vertices[FACE_INDICES[3][2]], color)
        }}
    {
        computeNormal(); // Compute face normals
    }

    // Create regular tetrahedron centered at origin with given radius
    static Tetrahedron createRegular(double radius, const ColorDBL& color) {
        const double a = radius * 2.0 / std::sqrt(6.0);
        const double h = radius * 3.0 / std::sqrt(6.0);
        
        std::array<Point3d, 4> vertices = {{
            Point3d(0, -radius, -h/3),             // Bottom center
            Point3d(-a/2, radius/3, -h/3),         // Back left
            Point3d(a/2, radius/3, -h/3),          // Back right
            Point3d(0, 0, 2*h/3)                   // Top
        }};

        return Tetrahedron(vertices, color);
    }

private:
    void initializeFaces() {
        // Initialize the four triangular faces
        for (size_t i = 0; i < 4; ++i) {
            faces_[i] = Triangle(
                vertices_[FACE_INDICES[i][0]],
                vertices_[FACE_INDICES[i][1]],
                vertices_[FACE_INDICES[i][2]],
                color_
            );
        }
    }

    void computeNormal() override {
        // Compute normals for each face
        for (size_t i = 0; i < 4; ++i) {
            Vec3d edge1 = vertices_[FACE_INDICES[i][1]] - vertices_[FACE_INDICES[i][0]];
            Vec3d edge2 = vertices_[FACE_INDICES[i][2]] - vertices_[FACE_INDICES[i][0]];
            face_normals_[i] = cross(edge1, edge2).normalized();
        }
    }

    // Check if point is inside tetrahedron using barycentric coordinates
    [[nodiscard]] bool containsPoint(const Point3d& point) const {
        for (size_t i = 0; i < 4; ++i) {
            if (dot(face_normals_[i], point - vertices_[FACE_INDICES[i][0]]) > 0) {
                return false;
            }
        }
        return true;
    }

public:
    [[nodiscard]] std::optional<Intersection> intersect(const Ray& ray) const override {
        // First, find the closest triangle intersection
        std::optional<Intersection> closest_hit;
        double min_distance = std::numeric_limits<double>::infinity();

        for (const auto& face : faces_) {
            if (auto hit = face.intersect(ray)) {
                if (hit->distance < min_distance) {
                    min_distance = hit->distance;
                    closest_hit = hit;
                }
            }
        }

        return closest_hit;
    }

    // Get the face normal for a given intersection point
    [[nodiscard]] Vec3d getFaceNormal(const Point3d& point) const {
        for (size_t i = 0; i < 4; ++i) {
            const auto& face = faces_[i];
            if (auto hit = face.intersect(Ray(point + face_normals_[i] * 0.001, -face_normals_[i]))) {
                if (hit->distance < 0.002) {
                    return face_normals_[i];
                }
            }
        }
        return Vec3d(0, 1, 0); // Fallback normal (should never happen)
    }

    // Get volume of the tetrahedron
    [[nodiscard]] double getVolume() const {
        Vec3d v01 = vertices_[1] - vertices_[0];
        Vec3d v02 = vertices_[2] - vertices_[0];
        Vec3d v03 = vertices_[3] - vertices_[0];
        return std::abs(dot(cross(v01, v02), v03)) / 6.0;
    }

    // Get surface area of the tetrahedron
    [[nodiscard]] double getSurfaceArea() const {
        double total_area = 0.0;
        for (const auto& face : faces_) {
            Vec3d edge1 = vertices_[1] - vertices_[0];
            Vec3d edge2 = vertices_[2] - vertices_[0];
            total_area += cross(edge1, edge2).length() / 2.0;
        }
        return total_area;
    }

    // Get centroid of the tetrahedron
    [[nodiscard]] Point3d getCentroid() const {
        return (vertices_[0] + vertices_[1] + vertices_[2] + vertices_[3]) / 4.0;
    }

    // Transform the tetrahedron
    void translate(const Vec3d& offset) {
        for (auto& vertex : vertices_) {
            vertex += offset;
        }
        initializeFaces();
        computeNormal();
    }

    void scale(double factor) {
        Point3d centroid = getCentroid();
        for (auto& vertex : vertices_) {
            vertex = centroid + (vertex - centroid) * factor;
        }
        initializeFaces();
        computeNormal();
    }

    void rotate(const Vec3d& axis, double angle) {
        // Quaternion rotation implementation
        double half_angle = angle * 0.5;
        double sin_half = std::sin(half_angle);
        
        Vec3d normalized_axis = axis.normalized();
        Vec3d q_vector = normalized_axis * sin_half;
        double q_scalar = std::cos(half_angle);
        
        Point3d centroid = getCentroid();
        for (auto& vertex : vertices_) {
            Vec3d v = vertex - centroid;
            
            // Apply quaternion rotation: q * v * q^-1
            Vec3d t = cross(q_vector, v) * 2.0;
            vertex = centroid + v + q_scalar * t + cross(q_vector, t);
        }
        
        initializeFaces();
        computeNormal();
    }

    // Add getter for vertices (needed for scene bounds calculation)
    [[nodiscard]] const std::array<Point3d, 4>& getVertices() const noexcept {
        return vertices_;
    }
}; 