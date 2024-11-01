#include "Scene.h"
#include "Camera.h"
#include "ImplicitObject.h"
#include "Polygon.h"
#include "Random.h"
#include "WindowX11.h"
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

void renderRegion(Camera& camera, const Scene& scene, 
                 uint32_t start_y, uint32_t end_y,
                 int samples_per_pixel) {
    std::vector<std::tuple<uint32_t, uint32_t, ColorDBL>> pixel_batch;
    pixel_batch.reserve(camera.getWidth());

    for (uint32_t y = start_y; y < end_y; ++y) {
        pixel_batch.clear();
        
        for (uint32_t x = 0; x < camera.getWidth(); ++x) {
            ColorDBL pixel_color(0, 0, 0);

            for (int s = 0; s < samples_per_pixel; ++s) {
                Ray ray = camera.getRayForPixel(x, y, true);
                if (auto hit = scene.intersect(ray)) {
                    // Start with ambient light to ensure everything is visible
                    ColorDBL ambient(0.1, 0.1, 0.1);
                    pixel_color += ambient;

                    if (hit->implicitObject && hit->implicitObject->isEmissive()) {
                        pixel_color += hit->emission;
                    } else {
                        // Add direct lighting
                        ColorDBL direct_light(0, 0, 0);
                        for (const auto& light_point : scene.sampleLights(1)) {
                            Vec3d light_dir = (light_point - hit->point).normalized();
                            double cos_theta = std::max(0.0, dot(hit->normal, light_dir));
                            
                            if (cos_theta > 0 && scene.isVisible(hit->point, light_point)) {
                                double light_distance = (light_point - hit->point).length();
                                double attenuation = 1.0 / (light_distance * light_distance);
                                direct_light += hit->color * cos_theta * attenuation * 50.0; // Increased intensity
                            }
                        }
                        pixel_color += direct_light;
                    }
                } else {
                    // Add background color for rays that don't hit anything
                    pixel_color += ColorDBL(0.2, 0.2, 0.2);
                }
            }

            // Average samples and apply gamma correction
            pixel_color = pixel_color / samples_per_pixel;
            pixel_color = ColorDBL(
                std::pow(std::clamp(pixel_color.r(), 0.0, 1.0), 1.0/2.2),
                std::pow(std::clamp(pixel_color.g(), 0.0, 1.0), 1.0/2.2),
                std::pow(std::clamp(pixel_color.b(), 0.0, 1.0), 1.0/2.2)
            );
            
            pixel_batch.emplace_back(x, y, pixel_color);
        }

        camera.setPixelBatch(pixel_batch);
    }
}

int main() {
    Random::init();
    Scene scene;

    // Create window first
    WindowX11 window(1280, 720, "Real-time Ray Tracer");

    // Cornell Box dimensions
    const double room_size = 4.0;
    const double half_size = room_size / 2.0;
    
    // Room walls setup (same as your original setup)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(0, 0, -half_size),
        Vec3d(0, 0, 1),
        ColorDBL(0.9, 0.9, 0.9)
    ));
    
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(-half_size, 0, 0),
        Vec3d(1, 0, 0),
        ColorDBL(0.9, 0.2, 0.2)
    ));
    
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(half_size, 0, 0),
        Vec3d(-1, 0, 0),
        ColorDBL(0.2, 0.9, 0.2)
    ));
    
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(0, -half_size, 0),
        Vec3d(0, 1, 0),
        ColorDBL(0.9, 0.9, 0.9)
    ));
    
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(0, half_size, 0),
        Vec3d(0, -1, 0),
        ColorDBL(0.9, 0.9, 0.9)
    ));

    // Camera settings for real-time viewing
    Camera::Settings camera_settings;
    camera_settings.width = window.getWidth();
    camera_settings.height = window.getHeight();
    camera_settings.fov = 90.0;  // Wider FOV for FPS-style viewing
    camera_settings.use_dof = false;

    // Initial camera position
    Camera camera(
        Point3d(0, 0, half_size * 0.8),  // Start position
        Point3d(0, 0, -1),               // Looking into the room
        Vec3d(0, 1, 0),                  // Up vector
        camera_settings
    );

    // Add objects (same as your original setup)
    scene.addImplicitObject(std::make_shared<Sphere>(
        Point3d(-1.0, -half_size + 0.8, -half_size + 1.0),
        0.8,
        ColorDBL(0.95, 0.95, 0.95)
    ));

    scene.addImplicitObject(std::make_shared<Sphere>(
        Point3d(1.0, -half_size + 0.4, -half_size + 1.0),
        0.4,
        ColorDBL(0.7, 0.3, 0.3)
    ));

    auto light = std::make_shared<Sphere>(
        Point3d(0, half_size - 0.1, 0),
        0.2,
        ColorDBL(1.0, 1.0, 1.0)
    );
    light->makeEmissive(ColorDBL(1.0, 1.0, 1.0), 100.0);
    scene.addImplicitObject(light);

    // Real-time rendering settings
    const int samples_per_frame = 1;  // Reduced samples for real-time
    double mouse_dx = 0, mouse_dy = 0;
    double yaw = 0, pitch = 0;
    const double mouse_sensitivity = 0.002;
    const double move_speed = 0.1;

    std::cout << "Controls:\n"
              << "Tab: Toggle mouse capture\n"
              << "WASD: Move camera\n"
              << "Mouse: Look around\n"
              << "Escape: Exit\n" << std::endl;

    // Main rendering loop
    while (window.processEvents(mouse_dx, mouse_dy)) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        // Update camera rotation
        yaw += mouse_dx * mouse_sensitivity;
        pitch = std::clamp(pitch - mouse_dy * mouse_sensitivity, 
                          -M_PI/2.0 + 0.1, M_PI/2.0 - 0.1);

        // Update camera orientation
        Vec3d forward(
            std::cos(pitch) * std::cos(yaw),
            std::sin(pitch),
            std::cos(pitch) * std::sin(yaw)
        );
        
        camera.lookAt(camera.getPosition(), camera.getPosition() + forward, Vec3d(0, 1, 0));

        // Update camera position based on keyboard input
        Vec3d right = cross(forward, Vec3d(0, 1, 0)).normalized();
        Point3d pos = camera.getPosition();
        
        if (window.isKeyPressed(XK_w)) {
            pos = pos + forward * move_speed;
        }
        if (window.isKeyPressed(XK_s)) {
            pos = pos - forward * move_speed;
        }
        if (window.isKeyPressed(XK_a)) {
            pos = pos - right * move_speed;
        }
        if (window.isKeyPressed(XK_d)) {
            pos = pos + right * move_speed;
        }
        if (window.isKeyPressed(XK_space)) {
            pos = pos + Vec3d(0, 1, 0) * move_speed;
        }
        if (window.isKeyPressed(XK_Control_L)) {
            pos = pos - Vec3d(0, 1, 0) * move_speed;
        }

        camera.lookAt(pos, pos + forward, Vec3d(0, 1, 0));

        // Reset progress for new frame
        camera.resetProgress();

        // Render frame
        const uint32_t num_threads = std::thread::hardware_concurrency();
        const uint32_t rows_per_thread = camera.getHeight() / num_threads;

        std::vector<std::thread> threads;
        for (uint32_t i = 0; i < num_threads; ++i) {
            uint32_t start_row = i * rows_per_thread;
            uint32_t end_row = (i == num_threads - 1) ? camera.getHeight() : (i + 1) * rows_per_thread;
            threads.emplace_back(renderRegion, std::ref(camera), std::ref(scene), 
                               start_row, end_row, samples_per_frame);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Update display
        window.updateScreen(camera.getImageBuffer());

        // Calculate frame time
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        
        // Display FPS in window title
        std::string title = "Real-time Ray Tracer - FPS: " + std::to_string(1000.0 / frame_time);
        window.setTitle(title);
    }

    return 0;
}