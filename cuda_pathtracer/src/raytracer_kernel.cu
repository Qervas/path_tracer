#include "raytracer_kernel.cuh"
#include "Camera.cuh"
#include "Scene.cuh"
#include "Error.cuh"

// Define device constants
__constant__ GPUCamera d_camera;
__constant__ GPUSphere d_spheres[16];
__constant__ GPUPlane d_planes[16];
__constant__ int d_num_spheres;
__constant__ int d_num_planes;

// Device intersection structure
struct GPUIntersection {
    float3 point;
    float3 normal;
    float distance;
    float3 color;
    bool frontFace;
    float3 emission;
    bool hit;

    // Add conversion constructor from Intersection_t
    __device__ GPUIntersection& operator=(const Intersection_t& isect) {
        point = make_float3(isect.point.x, isect.point.y, isect.point.z);
        normal = make_float3(isect.normal.x, isect.normal.y, isect.normal.z);
        distance = isect.distance;
        color = make_float3(isect.color.r, isect.color.g, isect.color.b);
        frontFace = isect.frontFace;
        emission = make_float3(isect.emission.r, isect.emission.g, isect.emission.b);
        hit = isect.hit;
        return *this;
    }
};

// Device helper functions
__device__ Intersection_t intersectSphere(const Ray_t& ray, const GPUSphere& sphere) {
    Intersection_t isect;
    isect.hit = false;
    
    Vec3f_t sphere_center = Vec3f_t::fromFloat3(sphere.center);
    Vec3f_t oc = ray.origin - sphere_center;
    
    float a = 1.0f;  // Optimized since ray direction is normalized
    float half_b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = half_b * half_b - a * c;
    
    if (discriminant < 0) return isect;
    
    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;
    
    if (root < 0.001f || root > 1000.0f) {
        root = (-half_b + sqrtd) / a;
        if (root < 0.001f || root > 1000.0f) return isect;
    }
    
    isect.hit = true;
    isect.distance = root;
    isect.point = ray.at(root);
    isect.normal = (isect.point - sphere_center) / sphere.radius;
    isect.color = Color_t::fromFloat3(sphere.color);
    isect.emission = sphere.is_emissive ? Color_t::fromFloat3(sphere.emission) : Color_t(0.0f, 0.0f, 0.0f);
    isect.frontFace = dot(ray.direction, isect.normal) < 0;
    if (!isect.frontFace) isect.normal = -isect.normal;
    
    return isect;
}

__device__ Intersection_t intersectPlane(const Ray_t& ray, const GPUPlane& plane) {
    Intersection_t isect;
    isect.hit = false;
    
    Vec3f_t plane_normal = Vec3f_t::fromFloat3(plane.normal);
    Vec3f_t plane_point = Vec3f_t::fromFloat3(plane.point);
    
    float denom = dot(plane_normal, ray.direction);
    if (fabsf(denom) < 1e-8f) return isect;
    
    float t = dot(plane_point - ray.origin, plane_normal) / denom;
    if (t < 0.001f || t > 1000.0f) return isect;
    
    isect.hit = true;
    isect.distance = t;
    isect.point = ray.at(t);
    isect.normal = plane_normal;
    isect.color = Color_t::fromFloat3(plane.color);
    isect.emission = Color_t(0.0f, 0.0f, 0.0f);
    isect.frontFace = dot(ray.direction, plane_normal) < 0;
    if (!isect.frontFace) isect.normal = -plane_normal;
    
    return isect;
}

// Add these helper functions at the top of the file
__device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__global__ void renderKernel(float4* output, uint32_t width, uint32_t height, uint32_t frame_count) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t pixel_index = y * width + x;
    
    // Initialize random state
    curandState rand_state;
    curand_init(pixel_index + frame_count * width * height, 0, 0, &rand_state);
    
    // Calculate UV coordinates
    float u = (x + curand_uniform(&rand_state)) / static_cast<float>(width);
    float v = (y + curand_uniform(&rand_state)) / static_cast<float>(height);
    
    // Generate ray from camera
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    float viewport_height = 2.0f * tanf(d_camera.fov / 2.0f);
    float viewport_width = aspect * viewport_height;
    
    Vec3f_t horizontal = Vec3f_t::fromFloat3(d_camera.right) * viewport_width;
    Vec3f_t vertical = Vec3f_t::fromFloat3(d_camera.up) * viewport_height;
    Vec3f_t lower_left_corner = Vec3f_t::fromFloat3(d_camera.origin) + 
                               Vec3f_t::fromFloat3(d_camera.forward) - 
                               horizontal * 0.5f - vertical * 0.5f;
    
    Ray_t ray;
    ray.origin = Vec3f_t::fromFloat3(d_camera.origin);
    ray.direction = (lower_left_corner + horizontal * u + vertical * (1.0f - v) - ray.origin).normalized();
    
    // Initialize color
    Color_t pixel_color(0.0f, 0.0f, 0.0f);
    
    // Trace ray
    GPUIntersection isect;
    isect.hit = false;
    float closest_dist = FLOAT_MAX;
    
    // Check sphere intersections
    for (int i = 0; i < d_num_spheres; i++) {
        Intersection_t sphere_isect = intersectSphere(ray, d_spheres[i]);
        if (sphere_isect.hit && sphere_isect.distance < closest_dist) {
            closest_dist = sphere_isect.distance;
            isect = sphere_isect;  // This will now use our conversion operator
        }
    }
    
    // Check plane intersections
    for (int i = 0; i < d_num_planes; i++) {
        Intersection_t plane_isect = intersectPlane(ray, d_planes[i]);
        if (plane_isect.hit && plane_isect.distance < closest_dist) {
            closest_dist = plane_isect.distance;
            isect = plane_isect;  // This will now use our conversion operator
        }
    }
    
    // Calculate color based on intersection
    if (isect.hit) {
        pixel_color = isect.color * 0.5f + isect.emission;
    }
    
    // Accumulate samples if frame_count > 0
    if (frame_count > 0) {
        float4 prev_color = output[pixel_index];
        float t = 1.0f / (frame_count + 1);
        float3 current_color = make_float3(prev_color.x, prev_color.y, prev_color.z);
        pixel_color = pixel_color * t + current_color * (1.0f - t);
    }
    
    // Write final color
    output[pixel_index] = make_float4(pixel_color.r, pixel_color.g, pixel_color.b, 1.0f);
}

// Add initialization function
void initializeGPUData(const Camera_t& camera, const Scene_t* d_scene) {
    // Validate scene pointer
    if (!d_scene) {
        throw std::runtime_error("Scene pointer is null in initializeGPUData");
    }

    // Copy scene structure from device to host
    Scene_t h_scene;
    CUDA_CHECK(cudaMemcpy(&h_scene, d_scene, sizeof(Scene_t), cudaMemcpyDeviceToHost));

    // Setup camera data
    GPUCamera h_camera;
    h_camera.origin = make_float3(camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
    h_camera.forward = make_float3(camera.getForward().x, camera.getForward().y, camera.getForward().z);
    h_camera.right = make_float3(camera.getRight().x, camera.getRight().y, camera.getRight().z);
    h_camera.up = make_float3(camera.getUp().x, camera.getUp().y, camera.getUp().z);
    h_camera.fov = camera.getSettings().fov * M_PI / 180.0f;
    h_camera.width = camera.getWidth();
    h_camera.height = camera.getHeight();

    // Copy camera data to GPU
    cudaMemcpyToSymbol(d_camera, &h_camera, sizeof(GPUCamera));

    // Setup scene objects
    std::vector<GPUSphere> h_spheres;
    std::vector<GPUPlane> h_planes;

    // Validate implicit object count and pointer
    if (h_scene.implicit_object_count <= 0 || !h_scene.d_implicit_objects) {
        // If there are no objects, just set the counts to 0 and return
        int num_spheres = 0;
        int num_planes = 0;
        cudaMemcpyToSymbol(d_num_spheres, &num_spheres, sizeof(int));
        cudaMemcpyToSymbol(d_num_planes, &num_planes, sizeof(int));
        return;
    }

    // Create host array of implicit objects
    ImplicitObject_t** h_implicit_objects = nullptr;
    try {
        h_implicit_objects = new ImplicitObject_t*[h_scene.implicit_object_count];
        
        // Copy device pointers to host
        CUDA_CHECK(cudaMemcpy(h_implicit_objects, h_scene.d_implicit_objects, 
                             h_scene.implicit_object_count * sizeof(ImplicitObject_t*), 
                             cudaMemcpyDeviceToHost));

        // Convert implicit objects to GPU format
        for (uint32_t i = 0; i < h_scene.implicit_object_count; ++i) {
            const ImplicitObject_t* obj = h_implicit_objects[i];
            
            // Check if object is a sphere by calling isSphere()
            if (obj->isSphere()) {
                GPUSphere gpu_sphere;
                gpu_sphere.center = make_float3(obj->getCenter().x, obj->getCenter().y, obj->getCenter().z);
                gpu_sphere.radius = obj->getRadius();
                gpu_sphere.color = make_float3(obj->getColor().r, obj->getColor().g, obj->getColor().b);
                gpu_sphere.is_emissive = obj->isEmissive();
                if (obj->isEmissive()) {
                    Color_t emission = obj->getEmissionColor() * obj->getEmissionStrength();
                    gpu_sphere.emission = make_float3(emission.r, emission.g, emission.b);
                } else {
                    gpu_sphere.emission = make_float3(0.0f, 0.0f, 0.0f);
                }
                h_spheres.push_back(gpu_sphere);
            }
        }

        // Cleanup host array
        delete[] h_implicit_objects;

        // Copy scene data to GPU
        int num_spheres = static_cast<int>(h_spheres.size());
        cudaMemcpyToSymbol(d_num_spheres, &num_spheres, sizeof(int));
        if (num_spheres > 0) {
            cudaMemcpyToSymbol(d_spheres, h_spheres.data(), h_spheres.size() * sizeof(GPUSphere));
        }

        int num_planes = 0;  // We'll add plane support later if needed
        cudaMemcpyToSymbol(d_num_planes, &num_planes, sizeof(int));

    } catch (const std::exception& e) {
        if (h_implicit_objects) {
            delete[] h_implicit_objects;
        }
        throw;
    }
}

// Add this function to handle kernel launch
extern "C" void launchRenderKernel(float4* output, uint32_t width, uint32_t height, uint32_t frame_count, dim3 grid, dim3 block) {
    renderKernel<<<grid, block>>>(output, width, height, frame_count);
} 