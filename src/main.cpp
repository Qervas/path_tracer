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

std::mutex cout_mutex;  // Global mutex for cout synchronization

// Add progress monitoring thread function
void monitorProgress(const Camera& camera, std::atomic<bool>& rendering) {
    while (rendering) {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "\rProgress: " 
                     << static_cast<int>(camera.getProgress() * 100) << "%" 
                     << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

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
                    ColorDBL direct_light(0, 0, 0);
                    
                    // Add emission if the object is emissive
                    if (hit->implicitObject && hit->implicitObject->isEmissive()) {
                        pixel_color += hit->emission;
                        continue;  // Skip direct lighting for emissive objects
                    }

                    // Sample points on light sources
                    for (const auto& light_point : scene.sampleLights(8)) {
                        Vec3d light_dir = (light_point - hit->point).normalized();
                        double cos_theta = std::max(0.0, dot(hit->normal, light_dir));
                        
                        // Check visibility
                        if (cos_theta > 0 && scene.isVisible(hit->point, light_point)) {
                            // Calculate light attenuation with more realistic falloff
                            double light_distance = (light_point - hit->point).length();
                            double attenuation = 1.0 / (1.0 + light_distance * light_distance);
                            
                            direct_light += hit->color * cos_theta * attenuation * 20.0;  // Reduced intensity
                        }
                    }
                    
                    pixel_color += direct_light;
                }
            }

            pixel_color = pixel_color / samples_per_pixel;
            
            // Apply gamma correction
            pixel_color = ColorDBL(
                std::pow(pixel_color.r(), 1.0/2.2),
                std::pow(pixel_color.g(), 1.0/2.2),
                std::pow(pixel_color.b(), 1.0/2.2)
            );
            
            pixel_batch.emplace_back(x, y, pixel_color);
        }

        camera.setPixelBatch(pixel_batch);
        camera.incrementProgress(camera.getWidth());
    }
}

int main() {
    Random::init();
    Scene scene;

    // Cornell Box dimensions - adjust for better proportions
    const double room_size = 4.0;  // Slightly smaller for better view
    const double half_size = room_size / 2.0;
    
    // Room walls with higher reflectivity
    // Back wall (white)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(0, 0, -half_size),
        Vec3d(0, 0, 1),
        ColorDBL(0.9, 0.9, 0.9)  // Brighter white
    ));
    
    // Left wall (red)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(-half_size, 0, 0),
        Vec3d(1, 0, 0),
        ColorDBL(0.9, 0.2, 0.2)  // Brighter red
    ));
    
    // Right wall (green)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(half_size, 0, 0),
        Vec3d(-1, 0, 0),
        ColorDBL(0.2, 0.9, 0.2)  // Brighter green
    ));
    
    // Floor (white)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(0, -half_size, 0),
        Vec3d(0, 1, 0),
        ColorDBL(0.9, 0.9, 0.9)
    ));
    
    // Ceiling (white)
    scene.addImplicitObject(std::make_shared<Plane>(
        Point3d(0, half_size, 0),
        Vec3d(0, -1, 0),
        ColorDBL(0.9, 0.9, 0.9)
    ));

    // Camera setup with classic Cornell box view
    Camera::Settings camera_settings;
    camera_settings.width = 800;
    camera_settings.height = 800;  // Make it square like classic Cornell box
    camera_settings.fov = 45.0;     // Classic narrow FOV for Cornell box
    camera_settings.use_dof = false;

    // Position camera at the back wall, looking forward
    Camera camera(
        Point3d(0, 0, -half_size),    // origin
        Point3d(0, -half_size * 0.99, 0),                 // target
        Vec3d(0, 1, 0),                    // Up vector
        camera_settings
    );

    // Adjust sphere positions for classic view
    // Large metallic sphere
    scene.addImplicitObject(std::make_shared<Sphere>(
        Point3d(-1.0, -half_size + 0.8, -half_size + 1.0),  // Left tall sphere
        0.8,
        ColorDBL(0.95, 0.95, 0.95)  // Metallic white
    ));

    // Smaller colored sphere
    scene.addImplicitObject(std::make_shared<Sphere>(
        Point3d(1.0, -half_size + 0.4, -half_size + 1.0),   // Right short sphere
        0.4,
        ColorDBL(0.7, 0.3, 0.3)  // Reddish
    ));

    // Light source - centered near ceiling
    auto light = std::make_shared<Sphere>(
        Point3d(0, half_size - 0.1, 0),  // Centered light
        0.2,  // Slightly larger for better illumination
        ColorDBL(1.0, 1.0, 1.0)
    );
    light->makeEmissive(ColorDBL(1.0, 1.0, 1.0), 25.0);  // Increased brightness a bit
    scene.addImplicitObject(light);

    // Render settings
    const int samples_per_pixel = 100;  // More samples for less noise
    
    std::cout << "Rendering at " << camera_settings.width << "x" << camera_settings.height 
              << " with " << samples_per_pixel << " samples per pixel..." << std::endl;

    // Reset progress
    camera.resetProgress();

    // Progress tracking
    std::atomic<bool> rendering{true};
    std::thread progress_thread(monitorProgress, std::ref(camera), std::ref(rendering));

    // Calculate number of threads and rows per thread
    const uint32_t num_threads = std::thread::hardware_concurrency();
    const uint32_t rows_per_thread = camera.getHeight() / num_threads;

    // Create and start threads
    std::vector<std::thread> threads;
    std::cout << "Rendering using " << num_threads << " threads..." << std::endl;

    for (uint32_t i = 0; i < num_threads; ++i) {
        uint32_t start_row = i * rows_per_thread;
        uint32_t end_row = (i == num_threads - 1) ? camera.getHeight() : (i + 1) * rows_per_thread;
        threads.emplace_back(renderRegion, std::ref(camera), std::ref(scene), 
                           start_row, end_row, samples_per_pixel);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Stop progress monitoring
    rendering = false;
    progress_thread.join();

    std::cout << "\nSaving image..." << std::endl;
    camera.saveImage("cornell_box_hd.ppm");
    std::cout << "Done!" << std::endl;

    return 0;
}