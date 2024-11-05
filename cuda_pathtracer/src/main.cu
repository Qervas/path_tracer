#include <cuda_runtime.h>
#include "Window.cuh"
#include "Scene.cuh"
#include "Camera.cuh"
#include "Random.cuh"
#include "raytracer_kernel.cuh"
#include "Material.cuh"
#include "MaterialFactory.cuh"
#include <iostream>
#include <chrono>
#include <algorithm>
#include "Error.cuh"

void checkCudaCapabilities() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw "No CUDA capable devices found";
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
}

void initializeCuda() {
    // Initialize CUDA runtime
    CUDA_CHECK(cudaSetDevice(0));
    
    // Create context
    cudaFree(0);
    
    // Check last error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw cudaGetErrorString(error);
    }
}

int main() {
    try {
        // Initialize CUDA with proper error checking
        std::cout << "Initializing CUDA..." << std::endl;
        checkCudaCapabilities();
        initializeCuda();
        
        // Create window
        std::cout << "Creating window..." << std::endl;
        Window_t window(1280, 720, "CUDA Ray Tracer");

        // Allocate CUDA memory with error checking
        std::cout << "Allocating GPU memory..." << std::endl;
        float4* d_output = nullptr;
        size_t required_memory = window.getWidth() * window.getHeight() * sizeof(float4);
        CUDA_CHECK(cudaMalloc(&d_output, required_memory));
        
        if (d_output == nullptr) {
            throw "Failed to allocate CUDA memory";
        }

        // Cornell Box dimensions
        const float room_size = 4.0f;
        const float half_size = room_size / 2.0f;

        // Create camera settings struct
        Camera_t::Settings camera_settings{
            90.0f,  // vertical FOV
            window.getWidth() / static_cast<float>(window.getHeight()),  // aspect ratio
            0.0f,   // aperture
            10.0f,  // focus distance
            window.getWidth(),
            window.getHeight(),
            false   // enable_dof
        };

        // Initialize camera
        Camera_t camera(
            Point3f_t(0, 0, half_size * 0.8f),  // position
            Point3f_t(0, 0, -1),                // look at
            Vec3f_t(0, 1, 0),                   // up vector
            camera_settings
        );

        // Create materials on device
        Material_t *d_white_diffuse = nullptr, *d_red_diffuse = nullptr, 
                  *d_green_diffuse = nullptr, *d_light = nullptr, 
                  *d_glass = nullptr, *d_metal = nullptr;
        
        MaterialFactory::createMaterialsOnDevice(
            &d_white_diffuse, &d_red_diffuse, &d_green_diffuse,
            &d_light, &d_glass, &d_metal
        );

        // Create scene objects
        SceneManager scene_manager;

        // Add walls
        scene_manager.addObject(new Plane_t(Point3f_t(0.0f, -2.0f, 0.0f), 
            Vec3f_t(0.0f, 1.0f, 0.0f), d_white_diffuse));
        scene_manager.addObject(new Plane_t(Point3f_t(0.0f, 2.0f, 0.0f), 
            Vec3f_t(0.0f, -1.0f, 0.0f), d_white_diffuse));
        scene_manager.addObject(new Plane_t(Point3f_t(0.0f, 0.0f, -2.0f), 
            Vec3f_t(0.0f, 0.0f, 1.0f), d_white_diffuse));
        scene_manager.addObject(new Plane_t(Point3f_t(-2.0f, 0.0f, 0.0f), 
            Vec3f_t(1.0f, 0.0f, 0.0f), d_red_diffuse));
        scene_manager.addObject(new Plane_t(Point3f_t(2.0f, 0.0f, 0.0f), 
            Vec3f_t(-1.0f, 0.0f, 0.0f), d_green_diffuse));

        // Add spheres
        auto glass_sphere = new Sphere_t(Point3f_t(-0.5f, -1.0f, -1.0f), 
            0.5f, d_glass);
        auto metal_sphere = new Sphere_t(Point3f_t(0.5f, -1.0f, 0.0f), 
            0.5f, d_metal);
        
        scene_manager.addObject(glass_sphere);
        scene_manager.addObject(metal_sphere);

        // Add light
        auto light_sphere = new Sphere_t(Point3f_t(0.0f, 1.9f, 0.0f), 
            0.5f, d_light);
        light_sphere->makeEmissive(Color_t(1.0f), 15.0f);
        scene_manager.addObject(light_sphere);

        // Upload scene to GPU
        scene_manager.uploadToGPU();

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

        // Cleanup materials before exit
        MaterialFactory::cleanup(
            d_white_diffuse, d_red_diffuse, d_green_diffuse,
            d_light, d_glass, d_metal
        );

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
} 