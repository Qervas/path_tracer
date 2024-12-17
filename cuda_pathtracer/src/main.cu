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
        Window_t window(1920, 1080, "CUDA Ray Tracer");

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
            true   // enable_dof
        };

        // Initialize camera
        Camera_t camera(
            Point3f_t(0, 0, half_size * 0.8f),  // position
            Point3f_t(0, 0, -1),                // look at
            Vec3f_t(0, -1, 0),                   // up vector
            camera_settings
        );

        // Create materials on device
        Material_t *d_white_diffuse = nullptr, *d_red_diffuse = nullptr,
                  *d_green_diffuse = nullptr, *d_light = nullptr,
                  *d_glass = nullptr, *d_metal = nullptr, *d_glossy = nullptr,
                    *d_mirror = nullptr;


        MaterialFactory::createMaterialsOnDevice(
            &d_white_diffuse, &d_red_diffuse, &d_green_diffuse,
            &d_light, &d_glass, &d_metal, &d_glossy, &d_mirror
        );

        // Create scene objects
        SceneManager scene_manager;

        // Room walls with subtle colors
        scene_manager.addObject(new Plane_t(Point3f_t(0.0f, -2.0f, 0.0f),
            Vec3f_t(0.0f, 1.0f, 0.0f), d_white_diffuse));  // Floor
        scene_manager.addObject(new Plane_t(Point3f_t(0.0f, 2.0f, 0.0f),
            Vec3f_t(0.0f, -1.0f, 0.0f), d_white_diffuse)); // Ceiling

        scene_manager.addObject(new Plane_t(Point3f_t(-2.0f, 0.0f, 0.0f),
            Vec3f_t(1.0f, 0.0f, 0.0f), d_red_diffuse));    // Left wall
        scene_manager.addObject(new Plane_t(Point3f_t(2.0f, 0.0f, 0.0f),
            Vec3f_t(-1.0f, 0.0f, 0.0f), d_green_diffuse)); // Right wall
        scene_manager.addObject(new Plane_t(
            Point3f_t(0.0f, 0.0f, -2.0f),  // Back wall
            Vec3f_t(0.0f, 0.0f, 1.0f),
            d_mirror
        ));

        scene_manager.addObject(new Plane_t(
            Point3f_t(0.0f, 0.0f, 2.0f),   // Front wall
            Vec3f_t(0.0f, 0.0f, -1.0f),    // Note the opposite normal
            d_mirror
        ));
        // Three main spheres showcasing different materials
        // Center: Large glass sphere (pure specular)
        auto glass_sphere = new Sphere_t(Point3f_t(0.0f, -1.0f, 0.0f),
            0.7f, d_glass);
        scene_manager.addObject(glass_sphere);

        // Left: Glossy sphere (mixed specular/diffuse)
        auto glossy_sphere = new Sphere_t(Point3f_t(-1.2f, -1.2f, -0.5f),
            0.6f, d_glossy);
        scene_manager.addObject(glossy_sphere);

        // Right: Metal sphere (pure specular with color)
        auto metal_sphere = new Sphere_t(Point3f_t(1.2f, -1.2f, -0.5f),
            0.6f, d_metal);
        scene_manager.addObject(metal_sphere);

        // Add some smaller spheres for interaction
        float small_radius = 0.25f;
        auto small_glass = new Sphere_t(Point3f_t(0.0f, -1.4f, -1.2f),
            small_radius, d_glass);
        scene_manager.addObject(small_glass);

        auto small_glossy1 = new Sphere_t(Point3f_t(-0.6f, -1.4f, -1.0f),
            small_radius, d_glossy);
        scene_manager.addObject(small_glossy1);

        auto small_glossy2 = new Sphere_t(Point3f_t(0.6f, -1.4f, -1.0f),
            small_radius, d_glossy);
        scene_manager.addObject(small_glossy2);

        // Lighting setup for better material visualization
        // Main ceiling light
        auto main_light = new Sphere_t(Point3f_t(0.0f, 1.8f, 0.0f),
            0.3f, d_light);
        main_light->makeEmissive(Color_t(1.0f), 15.0f);  // Pure white light
        scene_manager.addObject(main_light);

        // Colored accent lights
        auto blue_light = new Sphere_t(Point3f_t(1.8f, 0.0f, -1.8f),
            0.15f, d_light);
        blue_light->makeEmissive(Color_t(0.2f, 0.2f, 1.0f), 10.0f);
        scene_manager.addObject(blue_light);

        auto orange_light = new Sphere_t(Point3f_t(-1.8f, 0.0f, -1.8f),
            0.15f, d_light);
        orange_light->makeEmissive(Color_t(1.0f, 0.5f, 0.0f), 10.0f);
        scene_manager.addObject(orange_light);

        float tetra_radius = 0.6f;  // Size of the tetrahedron
        Point3f_t tetra_vertices[4] = {
            Point3f_t(0, tetra_radius, 0),                    // Top
            Point3f_t(-tetra_radius, -tetra_radius, tetra_radius),   // Front left
            Point3f_t(tetra_radius, -tetra_radius, tetra_radius),    // Front right
            Point3f_t(0, -tetra_radius, -tetra_radius)        // Back
        };

        // Create and position the tetrahedron
        Vec3f_t tetra_position(-0.6f, 0.0f, -1.0f);
        for (int i = 0; i < 4; ++i) {
            tetra_vertices[i] += tetra_position;
        }

        //tetrahedron faces
        const uint8_t face_indices[4][3] = {
            {0, 2, 1},  // Bottom face
            {0, 1, 3},  // Front face
            {1, 2, 3},  // Right face
            {2, 0, 3}   // Left face
        };

        for (int i = 0; i < 4; ++i) {
            auto triangle = new Triangle_t(
                tetra_vertices[face_indices[i][0]],
                tetra_vertices[face_indices[i][1]],
                tetra_vertices[face_indices[i][2]],
                d_glass
            );
            scene_manager.addObject(triangle);
        }
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

        std::vector<GPUSphere> h_spheres;
        std::vector<GPUPlane> h_planes;
        std::vector<GPUTriangle> h_triangles;


        // Process all objects once
        for (const auto& obj : scene_manager.getHostScene().h_implicit_objects) {
            if (obj->isSphere()) {
                const Sphere_t* sphere = static_cast<const Sphere_t*>(obj);
                GPUSphere gpu_sphere;
                gpu_sphere.center = make_float3(sphere->getCenter().x, sphere->getCenter().y, sphere->getCenter().z);
                gpu_sphere.radius = sphere->getRadius();
                gpu_sphere.material = sphere->getMaterial();
                gpu_sphere.is_emissive = sphere->isEmissive();
                if (sphere->isEmissive()) {
                    Color_t emission = sphere->getEmissionColor() * sphere->getEmissionStrength();
                    gpu_sphere.emission = make_float3(emission.r, emission.g, emission.b);
                } else {
                    gpu_sphere.emission = make_float3(0.0f, 0.0f, 0.0f);
                }
                h_spheres.push_back(gpu_sphere);
            }
            else if (obj->isPlane()) {
                const Plane_t* plane = static_cast<const Plane_t*>(obj);
                GPUPlane gpu_plane;
                gpu_plane.point = make_float3(plane->getPoint().x, plane->getPoint().y, plane->getPoint().z);
                gpu_plane.normal = make_float3(plane->getNormal().x, plane->getNormal().y, plane->getNormal().z);
                gpu_plane.material = plane->getMaterial();
                h_planes.push_back(gpu_plane);
            }
        }

        for (const auto& obj : scene_manager.getHostScene().h_implicit_objects) {
            if (auto triangle = dynamic_cast<const Triangle_t*>(obj)) {
                GPUTriangle gpu_triangle;
                const auto vertices = triangle->getVertices();
                gpu_triangle.v0 = make_float3(vertices[0].x, vertices[0].y, vertices[0].z);
                gpu_triangle.v1 = make_float3(vertices[1].x, vertices[1].y, vertices[1].z);
                gpu_triangle.v2 = make_float3(vertices[2].x, vertices[2].y, vertices[2].z);
                Vec3f_t normal = triangle->getNormalAt(vertices[0]);
                gpu_triangle.normal = make_float3(normal.x, normal.y, normal.z);
                gpu_triangle.material = triangle->getMaterial();
                h_triangles.push_back(gpu_triangle);
            }
        }

        // Upload constants once - only do this once
        int num_spheres = static_cast<int>(h_spheres.size());
        CUDA_CHECK(cudaMemcpyToSymbol(d_num_spheres, &num_spheres, sizeof(int)));
        if (num_spheres > 0) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_spheres, h_spheres.data(), h_spheres.size() * sizeof(GPUSphere)));
        }

        int num_planes = static_cast<int>(h_planes.size());
        CUDA_CHECK(cudaMemcpyToSymbol(d_num_planes, &num_planes, sizeof(int)));
        if (num_planes > 0) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_planes, h_planes.data(), h_planes.size() * sizeof(GPUPlane)));
        }

        int num_triangles = static_cast<int>(h_triangles.size());
        CUDA_CHECK(cudaMemcpyToSymbol(d_num_triangles, &num_triangles, sizeof(int)));
        if (num_triangles > 0) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_triangles, h_triangles.data(),
                                         h_triangles.size() * sizeof(GPUTriangle)));
        }

        initializeGPUData(camera, scene_manager.getDeviceScene());
        // Main rendering loop
        uint32_t frame_count = 0;

        while (window.processEvents()) {
            auto frame_start = std::chrono::high_resolution_clock::now();
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

            std::string title = "CUDA Path Tracer - FPS: " + std::to_string(1000.0f / frame_time);
            window.setTitle(title.c_str());

            frame_count++;
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_output));

        // Cleanup materials before exit
        MaterialFactory::cleanup(
            d_white_diffuse, d_red_diffuse, d_green_diffuse,
            d_light, d_glass, d_metal, d_glossy, d_mirror
        );

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
