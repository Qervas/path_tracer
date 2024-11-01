#include "raytracer_kernel.cuh"
#include "Camera.cuh"
#include "Scene.cuh"

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

__global__ void renderKernel(float4* output, uint32_t width, uint32_t height, uint32_t frame_count) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t pixel_index = y * width + x;
    
    // Initialize random state
    curandState rand_state;
    curand_init(pixel_index + frame_count * width * height, 0, 0, &rand_state);
    
    // Camera ray calculation
}

// Add initialization function
void initializeGPUData(const Camera_t& camera, const Scene_t* scene) {
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

    // Convert implicit objects to GPU format
    for (uint32_t i = 0; i < scene->implicit_object_count; ++i) {
        const ImplicitObject_t* obj = scene->d_implicit_objects[i];
        
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

    // Copy scene data to GPU
    int num_spheres = static_cast<int>(h_spheres.size());
    cudaMemcpyToSymbol(d_num_spheres, &num_spheres, sizeof(int));
    if (num_spheres > 0) {
        cudaMemcpyToSymbol(d_spheres, h_spheres.data(), h_spheres.size() * sizeof(GPUSphere));
    }

    int num_planes = 0;  // We'll add plane support later if needed
    cudaMemcpyToSymbol(d_num_planes, &num_planes, sizeof(int));
}

// Add this function to handle kernel launch
extern "C" void launchRenderKernel(float4* output, uint32_t width, uint32_t height, uint32_t frame_count, dim3 grid, dim3 block) {
    renderKernel<<<grid, block>>>(output, width, height, frame_count);
} 