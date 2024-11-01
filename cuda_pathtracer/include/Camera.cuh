#pragma once

#include "Vec3.cuh"
#include "Ray.cuh"
#include "Color.cuh"
#include "Random.cuh"
#include <curand_kernel.h>

class Camera_t {
public:
    // Camera settings struct for easy configuration
    struct Settings {
        float fov{60.0f};           // Vertical field of view in degrees
        float aspect_ratio{16.0f/9.0f}; // Width/Height ratio
        float aperture{0.0f};        // Lens aperture for depth of field
        float focus_dist{10.0f};     // Focus distance for depth of field
        uint32_t width{1920};       // Image width in pixels
        uint32_t height{1080};      // Image height in pixels
        bool use_dof{false};        // Enable/disable depth of field
    };

private:
    // Camera parameters
    Point3f_t origin_;        // Camera position
    Point3f_t target_;        // Look-at point
    Vec3f_t up_;             // Up vector
    Settings settings_;       // Camera settings

    // Derived values
    Vec3f_t u_, v_, w_;      // Camera basis vectors
    Point3f_t top_left_;     // Top-left corner of virtual screen
    Vec3f_t horizontal_;     // Vector across screen
    Vec3f_t vertical_;       // Vector down screen
    float lens_radius_;      // Derived from aperture

    float pitch_{0}, yaw_{0};

public:
    __host__ Camera_t(const Point3f_t& origin, const Point3f_t& target, const Vec3f_t& up, const Settings& settings)
        : origin_(origin)
        , target_(target)
        , up_(up.normalized())
        , settings_(settings)
    {
        initialize();
    }

    // Only declare these methods here, implementation moves to .cu file
    __host__ __device__ Ray_t getRay(float s, float t, curandState* rand_state = nullptr, bool jitter = true) const;
    __host__ __device__ Ray_t getRayForPixel(uint32_t x, uint32_t y, curandState* rand_state = nullptr, bool jitter = true) const;

    // Camera movement methods
    __host__ void lookAt(const Point3f_t& target);
    __host__ void lookAt(const Point3f_t& position, const Point3f_t& target, const Vec3f_t& up);

    __host__ void setPosition(const Point3f_t& position);
    __host__ void moveForward(float distance);
    __host__ void moveRight(float distance);
    __host__ void moveUp(float distance);
    __host__ void rotateYaw(float angle);
    __host__ void rotatePitch(float angle);

    // Getters
    __host__ __device__ const Point3f_t& getPosition() const { return origin_; }
    __host__ __device__ const Point3f_t& getTarget() const { return target_; }
    __host__ __device__ const Settings& getSettings() const { return settings_; }
    __host__ __device__ uint32_t getWidth() const { return settings_.width; }
    __host__ __device__ uint32_t getHeight() const { return settings_.height; }
    __host__ __device__ Vec3f_t getForward() const { return -w_; }
    __host__ __device__ Vec3f_t getRight() const { return u_; }
    __host__ __device__ Vec3f_t getUp() const { return v_; }

private:
    __host__ void initialize();
    __host__ void updateRotation();
    __device__ Point3f_t randomInUnitDisk(curandState* rand_state) const;
    __device__ Ray_t getRayDevice(float s, float t, curandState* rand_state, bool jitter = true) const;
    __host__ Ray_t getRayHost(float s, float t, bool jitter = true) const;
}; 