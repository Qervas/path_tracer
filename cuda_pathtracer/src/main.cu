#include <cuda_runtime.h>
#include "Window.cuh"
#include "Scene.cuh"
#include "Camera.cuh"
#include "Random.cuh"
#include "raytracer_kernel.cuh"
#include <iostream>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    try {
        // Create window
        Window_t window(1280, 720, "CUDA Ray Tracer");

        // Cornell Box dimensions
        const float room_size = 4.0f;
        const float half_size = room_size / 2.0f;

        // Initialize scene
        Scene_t scene;
        std::vector<ImplicitObject_t*> objects;
        std::vector<ImplicitObject_t*> lights;

        // Room walls
        auto back_wall = new Plane_t(
            Point3f_t(0, 0, -half_size),
            Vec3f_t(0, 0, 1),
            Color_t(0.9f, 0.9f, 0.9f)
        );
        objects.push_back(back_wall);

        auto left_wall = new Plane_t(
            Point3f_t(-half_size, 0, 0),
            Vec3f_t(1, 0, 0),
            Color_t(0.9f, 0.2f, 0.2f)
        );
        objects.push_back(left_wall);

        auto right_wall = new Plane_t(
            Point3f_t(half_size, 0, 0),
            Vec3f_t(-1, 0, 0),
            Color_t(0.2f, 0.9f, 0.2f)
        );
        objects.push_back(right_wall);

        auto floor = new Plane_t(
            Point3f_t(0, -half_size, 0),
            Vec3f_t(0, 1, 0),
            Color_t(0.9f, 0.9f, 0.9f)
        );
        objects.push_back(floor);

        auto ceiling = new Plane_t(
            Point3f_t(0, half_size, 0),
            Vec3f_t(0, -1, 0),
            Color_t(0.9f, 0.9f, 0.9f)
        );
        objects.push_back(ceiling);

        // Spheres
        auto sphere1 = new Sphere_t(
            Point3f_t(-1.0f, -half_size + 0.8f, -half_size + 1.0f),
            0.8f,
            Color_t(0.95f, 0.95f, 0.95f)
        );
        objects.push_back(sphere1);

        auto sphere2 = new Sphere_t(
            Point3f_t(1.0f, -half_size + 0.4f, -half_size + 1.0f),
            0.4f,
            Color_t(0.7f, 0.3f, 0.3f)
        );
        objects.push_back(sphere2);

        // Light
        auto light = new Sphere_t(
            Point3f_t(0, half_size - 0.1f, 0),
            0.2f,
            Color_t(1.0f, 1.0f, 1.0f)
        );
        light->makeEmissive(Color_t(1.0f, 1.0f, 1.0f), 100.0f);
        objects.push_back(light);
        lights.push_back(light);

        // Initialize camera
        Camera_t camera(
            Point3f_t(0, 0, half_size * 0.8f),
            Point3f_t(0, 0, -1),
            Vec3f_t(0, 1, 0),
            Camera_t::Settings{
                90.0f,                    // FOV
                window.getWidth() / static_cast<float>(window.getHeight()), // aspect ratio
                0.0f,                     // aperture
                10.0f,                    // focus distance
                window.getWidth(),        // width
                window.getHeight(),       // height
                false                     // DOF
            }
        );

        // Initialize CUDA resources
        float4* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, window.getWidth() * window.getHeight() * sizeof(float4)));

        // Initialize scene manager
        SceneManager scene_manager;
        scene_manager.initializeScene(objects, std::vector<Polygon_t*>(), lights);

        // Initialize GPU data
        initializeGPUData(camera, scene_manager.getDeviceScene());

        // Setup CUDA grid and blocks
        dim3 block(16, 16);
        dim3 grid(
            (window.getWidth() + block.x - 1) / block.x,
            (window.getHeight() + block.y - 1) / block.y
        );

        // Rendering loop variables
        float mouse_dx = 0, mouse_dy = 0;
        float yaw = 0, pitch = 0;
        const float mouse_sensitivity = 0.002f;
        const float move_speed = 0.1f;
        uint32_t frame_count = 0;
        bool camera_moved = false;

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
                             -M_PI_2f + 0.1f, M_PI_2f - 0.1f);

            // Update camera orientation
            Vec3f_t forward(
                cosf(pitch) * cosf(yaw),
                sinf(pitch),
                cosf(pitch) * sinf(yaw)
            );

            camera.lookAt(camera.getPosition(), camera.getPosition() + forward, Vec3f_t(0, 1, 0));

            // Handle keyboard input for camera movement
            Vec3f_t right = cross(forward, Vec3f_t(0, 1, 0)).normalized();
            Point3f_t pos = camera.getPosition();

            if (window.isKeyPressed(XK_w)) pos = pos + forward * move_speed;
            if (window.isKeyPressed(XK_s)) pos = pos - forward * move_speed;
            if (window.isKeyPressed(XK_a)) pos = pos - right * move_speed;
            if (window.isKeyPressed(XK_d)) pos = pos + right * move_speed;
            if (window.isKeyPressed(XK_space)) pos = pos + Vec3f_t(0, 1, 0) * move_speed;
            if (window.isKeyPressed(XK_Control_L)) pos = pos - Vec3f_t(0, 1, 0) * move_speed;

            camera.lookAt(pos, pos + forward, Vec3f_t(0, 1, 0));

            // Check if camera moved
            camera_moved = mouse_dx != 0 || mouse_dy != 0 || 
                          window.isKeyPressed(XK_w) || window.isKeyPressed(XK_s) ||
                          window.isKeyPressed(XK_a) || window.isKeyPressed(XK_d) ||
                          window.isKeyPressed(XK_space) || window.isKeyPressed(XK_Control_L);

            if (camera_moved) {
                frame_count = 0;
                initializeGPUData(camera, scene_manager.getDeviceScene());
            }

            // Launch render kernel
            launchRenderKernel(d_output, window.getWidth(), window.getHeight(), 
                             frame_count, grid, block);
            CUDA_CHECK(cudaGetLastError());

            // Update window with rendered frame
            window.updateScreen(d_output);

            // Calculate and display FPS
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                frame_end - frame_start).count();
            
            std::string title = "CUDA Ray Tracer - FPS: " + std::to_string(1000.0f / frame_time);
            window.setTitle(title.c_str());

            frame_count++;
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_output));
        for (auto obj : objects) delete obj;

    } catch (const char* error) {
        std::cerr << "Error: " << error << std::endl;
        return 1;
    }

    return 0;
} 