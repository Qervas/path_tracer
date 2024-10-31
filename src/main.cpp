#include "Scene.h"
#include "Camera.h"
#include "ImplicitObject.h"
#include "Polygon.h"
#include "Random.h"
#include <iostream>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

int main() {
    // Initialize random number generator
    Random::init();

    // Create scene
    Scene scene;

    // Cornell Box dimensions
    const double room_size = 2.0;
    const double half_size = room_size / 2.0;
    
    // Room walls (all planes facing inward)
    // Back wall (white)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(0, 0, -half_size),
        Vec3d(0, 0, 1),
        ColorDBL(0.7, 0.7, 0.7)
    ));
    
    // Left wall (red)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(-half_size, 0, 0),
        Vec3d(1, 0, 0),
        ColorDBL(0.7, 0.2, 0.2)
    ));
    
    // Right wall (green)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(half_size, 0, 0),
        Vec3d(-1, 0, 0),
        ColorDBL(0.2, 0.7, 0.2)
    ));
    
    // Floor (white)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(0, -half_size, 0),
        Vec3d(0, 1, 0),
        ColorDBL(0.7, 0.7, 0.7)
    ));
    
    // Ceiling (white)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(0, half_size, 0),
        Vec3d(0, -1, 0),
        ColorDBL(0.7, 0.7, 0.7)
    ));

    // Light source (bright white sphere near ceiling)
    auto light = std::make_shared<Sphere>(
        Point3d(0, half_size - 0.5, 0),  // Near the ceiling
        0.5,
        ColorDBL(1.0, 1.0, 1.0)
    );
    light->makeEmissive(ColorDBL(1.0, 1.0, 1.0), 15.0);  // Increased emission strength
    scene.addImplicitObject(light);

    // Add some spheres in the room
    scene.addImplicitObject(std::make_shared<Sphere>(
        Point3d(-1.0, -half_size + 1.0, -1.0),
        1.0,
        ColorDBL(0.7, 0.7, 0.7)
    ));

    scene.addImplicitObject(std::make_shared<Sphere>(
        Point3d(1.5, -half_size + 0.5, 0.5),
        0.5,
        ColorDBL(0.7, 0.7, 0.7)
    ));

    // Camera setup
    Camera::Settings camera_settings;
    camera_settings.width = 800;         // Reduced resolution for faster testing
    camera_settings.height = 600;
    camera_settings.fov = 90.0;         // Wider FOV to see more of the room
    camera_settings.use_dof = false;

    // Position camera to view the Cornell box
    Camera camera(
        Point3d(0, 0, room_size * 1.5),  // Camera pulled back
        Point3d(0, 0, 0),                // Looking at center
        Vec3d(0, 1, 0),                  // Up vector
        camera_settings
    );

    // Render settings
    const int samples_per_pixel = 100;
    
    std::cout << "Rendering..." << std::endl;

    // Progress tracking
    std::atomic<int> completed_rows{0};
    std::mutex cout_mutex;

    // Thread function to render a range of rows
    auto renderRows = [&](uint32_t start_row, uint32_t end_row) {
        for (uint32_t y = start_row; y < end_row; ++y) {
            for (uint32_t x = 0; x < camera.getWidth(); ++x) {
                ColorDBL pixel_color(0, 0, 0);

                for (int s = 0; s < samples_per_pixel; ++s) {
                    Ray ray = camera.getRayForPixel(x, y, true);
                    
                    if (auto hit = scene.intersect(ray)) {
                        ColorDBL direct_light(0, 0, 0);
                        
                        // Sample points on light sources
                        for (const auto& light_point : scene.sampleLights(4)) {
                            if (scene.isVisible(hit->point, light_point)) {
                                Vec3d light_dir = (light_point - hit->point).normalized();
                                double cos_theta = std::max(0.0, dot(hit->normal, light_dir));
                                direct_light += hit->color * cos_theta * 0.5;
                            }
                        }
                        
                        pixel_color += direct_light;
                        
                        if (hit->implicitObject && hit->implicitObject->isEmissive()) {
                            pixel_color += hit->emission;
                        }
                    }
                }

                pixel_color = pixel_color / samples_per_pixel;
                camera.setPixel(x, y, pixel_color);
            }

            // Update progress
            int completed = ++completed_rows;
            if (completed % 10 == 0) {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "\rProgress: " << (completed * 100 / camera.getHeight()) << "%" << std::flush;
            }
        }
    };

    // Calculate number of threads and rows per thread
    const uint32_t num_threads = std::thread::hardware_concurrency();
    const uint32_t rows_per_thread = camera.getHeight() / num_threads;

    // Create and start threads
    std::vector<std::thread> threads;
    std::cout << "Rendering using " << num_threads << " threads..." << std::endl;

    for (uint32_t i = 0; i < num_threads; ++i) {
        uint32_t start_row = i * rows_per_thread;
        uint32_t end_row = (i == num_threads - 1) ? camera.getHeight() : (i + 1) * rows_per_thread;
        threads.emplace_back(renderRows, start_row, end_row);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "\nSaving image..." << std::endl;
    camera.saveImage("cornell_box.ppm");
    std::cout << "Done!" << std::endl;

    return 0;
}