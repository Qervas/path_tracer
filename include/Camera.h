#pragma once

#include "Ray.h"
#include "Vec3.h"
#include "Color.h"
#include "Random.h"
#include <vector>
#include <memory>
#include <cmath>
#include <fstream>
#include <string>

class Camera {
public:
    // Camera settings struct for easy configuration
    struct Settings {
        double fov{60.0};              // Vertical field of view in degrees
        double aspect_ratio{16.0/9.0};  // Width/Height ratio
        double aperture{0.0};           // Lens aperture for depth of field (0 = pinhole)
        double focus_dist{10.0};        // Focus distance for depth of field
        uint32_t width{1920};          // Image width in pixels
        uint32_t height{1080};         // Image height in pixels
        bool use_dof{false};           // Enable/disable depth of field
    };

private:
    // Camera parameters
    Point3d origin_;          // Camera position
    Point3d target_;          // Look-at point
    Vec3d up_;               // Up vector
    Settings settings_;       // Camera settings

    // Derived values
    Vec3d u_, v_, w_;        // Camera basis vectors
    Point3d top_left_;       // Top-left corner of virtual screen
    Vec3d horizontal_;       // Vector across screen
    Vec3d vertical_;         // Vector down screen
    double lens_radius_;     // Derived from aperture

    // Image buffer
    std::vector<ColorDBL> image_buffer_;

public:
    Camera(const Point3d& origin, const Point3d& target, const Vec3d& up, const Settings& settings)
        : origin_(origin)
        , target_(target)
        , up_(up.normalized())
        , settings_(settings)
        , image_buffer_(settings.width * settings.height)
    {
        initialize();
    }

private:
    void initialize() {
        // Calculate camera basis vectors
        w_ = (origin_ - target_).normalized();  // Forward
        u_ = cross(up_, w_).normalized();       // Right
        v_ = cross(w_, u_);                     // Up

        // Calculate viewport dimensions
        double theta = settings_.fov * M_PI / 180.0;
        double viewport_height = 2.0 * std::tan(theta/2.0);
        double viewport_width = viewport_height * settings_.aspect_ratio;

        // Calculate virtual screen vectors
        horizontal_ = u_ * viewport_width * settings_.focus_dist;
        vertical_ = v_ * viewport_height * settings_.focus_dist;

        // Calculate top-left corner of virtual screen
        top_left_ = origin_ - horizontal_/2.0 + vertical_/2.0 - w_ * settings_.focus_dist;

        // Calculate lens radius for depth of field
        lens_radius_ = settings_.aperture / 2.0;
    }

    // Generate a random point in the unit disk for depth of field
    [[nodiscard]] Point3d randomInUnitDisk() const {
        while (true) {
            Point3d p(
                2.0 * Random::get() - 1.0,
                2.0 * Random::get() - 1.0,
                0
            );
            if (p.length_squared() < 1.0) return p;
        }
    }

public:
    // Generate a ray for a given pixel coordinate with optional jittering
    [[nodiscard]] Ray getRay(double s, double t, bool jitter = true) const {
        Point3d ray_origin = origin_;
        
        if (settings_.use_dof) {
            // Generate ray origin on lens for depth of field
            Point3d rd = randomInUnitDisk() * lens_radius_;
            Point3d offset = u_ * rd.x + v_ * rd.y;
            ray_origin = origin_ + offset;
        }

        if (jitter) {
            // Add random jitter for anti-aliasing
            s += Random::get() - 0.5;
            t += Random::get() - 0.5;
        }

        // Calculate point on virtual screen
        Point3d screen_point = top_left_ + horizontal_ * s - vertical_ * t;
        
        return Ray(ray_origin, (screen_point - ray_origin).normalized());
    }

    // Get ray for specific pixel
    [[nodiscard]] Ray getRayForPixel(uint32_t x, uint32_t y, bool jitter = true) const {
        double u = static_cast<double>(x) / (settings_.width - 1);
        double v = static_cast<double>(y) / (settings_.height - 1);
        return getRay(u, v, jitter);
    }

    // Set color for a specific pixel
    void setPixel(uint32_t x, uint32_t y, const ColorDBL& color) {
        if (x < settings_.width && y < settings_.height) {
            image_buffer_[y * settings_.width + x] = color;
        }
    }

    // Get color for a specific pixel
    [[nodiscard]] ColorDBL getPixel(uint32_t x, uint32_t y) const {
        if (x < settings_.width && y < settings_.height) {
            return image_buffer_[y * settings_.width + x];
        }
        return ColorDBL();
    }

    // Convert and save the image to a file
    void saveImage(const std::string& filename) const {
        // Create header for PPM format
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        out << "P6\n" << settings_.width << " " << settings_.height << "\n255\n";

        // Write pixel data
        for (const auto& color : image_buffer_) {
            // Apply gamma correction and convert to 8-bit color
            ColorRGB pixel(color.gamma(2.2));
            out.put(static_cast<char>(pixel.r));
            out.put(static_cast<char>(pixel.g));
            out.put(static_cast<char>(pixel.b));
        }

        out.close();
    }

    // Camera movement methods
    void lookAt(const Point3d& target) {
        target_ = target;
        initialize();
    }

    void setPosition(const Point3d& position) {
        origin_ = position;
        initialize();
    }

    void moveForward(double distance) {
        origin_ -= w_ * distance;
        initialize();
    }

    void moveRight(double distance) {
        origin_ += u_ * distance;
        initialize();
    }

    void moveUp(double distance) {
        origin_ += v_ * distance;
        initialize();
    }

    // Rotate camera around its axes
    void rotateYaw(double angle) {
        // Rotate around world up vector
        Vec3d forward = -w_;
        double cos_angle = std::cos(angle);
        double sin_angle = std::sin(angle);
        
        forward = Vec3d(
            forward.x * cos_angle - forward.z * sin_angle,
            forward.y,
            forward.x * sin_angle + forward.z * cos_angle
        );

        target_ = origin_ + forward;
        initialize();
    }

    void rotatePitch(double angle) {
        // Rotate around local right vector
        Vec3d forward = -w_;
        double cos_angle = std::cos(angle);
        double sin_angle = std::sin(angle);
        
        Vec3d new_forward = forward * cos_angle + v_ * sin_angle;
        target_ = origin_ + new_forward;
        initialize();
    }

    // Getters
    [[nodiscard]] const Point3d& getPosition() const noexcept { return origin_; }
    [[nodiscard]] const Point3d& getTarget() const noexcept { return target_; }
    [[nodiscard]] const Settings& getSettings() const noexcept { return settings_; }
    [[nodiscard]] uint32_t getWidth() const noexcept { return settings_.width; }
    [[nodiscard]] uint32_t getHeight() const noexcept { return settings_.height; }
}; 