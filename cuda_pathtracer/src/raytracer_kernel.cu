#include "raytracer_kernel.cuh"
#include "Camera.cuh"
#include "Scene.cuh"
#include "Error.cuh"
#include "Material.cuh"
#include "Intersection.cuh"
#include "ScatterRecord.cuh"
#include <curand_kernel.h>
#include <math_constants.h>

// Define device constants
__constant__ GPUCamera d_camera;
__constant__ GPUSphere d_spheres[32];
__constant__ GPUPlane d_planes[32];
__constant__ GPUTriangle d_triangles[128];
__constant__ int d_num_spheres;
__constant__ int d_num_planes;
__constant__ int d_num_triangles;

// Device intersection structure
struct GPUIntersection {
    float3 point;
    float3 normal;
    float distance;
    float3 color;
    bool frontFace;
    float3 emission;
    bool hit;

    // conversion constructor from Intersection_t
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

    if (!ray.isValidDistance(root)) {
        root = (-half_b + sqrtd) / a;
        if (!ray.isValidDistance(root)) return isect;
    }

    isect.hit = true;
    isect.distance = root;
    isect.point = ray.at(root);
    isect.normal = (isect.point - sphere_center) / sphere.radius;
    isect.material = sphere.material;
    isect.setFaceNormal(ray, isect.normal);

    if (sphere.is_emissive) {
        isect.emission = Color_t::fromFloat3(sphere.emission);
    }

    return isect;
}

__device__ Intersection_t intersectPlane(const Ray_t& ray, const GPUPlane& plane) {
    Intersection_t isect;
    isect.hit = false;

    Vec3f_t plane_normal = Vec3f_t::fromFloat3(plane.normal);
    Vec3f_t plane_point = Vec3f_t::fromFloat3(plane.point);

    float denom = dot(plane_normal, ray.direction);

    if (fabsf(denom) < 1e-6f) return isect;

    float t = dot(plane_point - ray.origin, plane_normal) / denom;

    if (!ray.isValidDistance(t)) return isect;

    isect.hit = true;
    isect.distance = t;
    isect.point = ray.at(t);
    isect.normal = plane_normal;
    isect.material = plane.material;
    isect.setFaceNormal(ray, plane_normal);

    return isect;
}


__device__ Intersection_t intersectTriangle(const Ray_t& ray, const GPUTriangle& triangle) {
    Intersection_t hit;
    hit.hit = false;

    // Möller–Trumbore intersection algorithm
    Vec3f_t v0 = Vec3f_t::fromFloat3(triangle.v0);
    Vec3f_t v1 = Vec3f_t::fromFloat3(triangle.v1);
    Vec3f_t v2 = Vec3f_t::fromFloat3(triangle.v2);

    Vec3f_t edge1 = v1 - v0;
    Vec3f_t edge2 = v2 - v0;
    Vec3f_t h = cross(ray.direction, edge2);
    float a = dot(edge1, h);

    // Check if ray is parallel to triangle
    if (fabsf(a) < 1e-8f)
        return hit;

    float f = 1.0f / a;
    Vec3f_t s = ray.origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return hit;

    Vec3f_t q = cross(s, edge1);
    float v = f * dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f)
        return hit;

    float t = f * dot(edge2, q);

    if (!ray.isValidDistance(t))
        return hit;

    hit.hit = true;
    hit.distance = t;
    hit.point = ray.at(t);
    hit.normal = Vec3f_t::fromFloat3(triangle.normal);
    hit.material = triangle.material;
    hit.setFaceNormal(ray, hit.normal);

    return hit;
}

__device__ Intersection_t intersectScene(const Ray_t& ray) {
    Intersection_t closest_hit;
    closest_hit.hit = false;
    float closest_dist = FLOAT_MAX;

    // Check sphere intersections
    for (int i = 0; i < d_num_spheres; i++) {
        Intersection_t sphere_isect = intersectSphere(ray, d_spheres[i]);
        if (sphere_isect.hit && sphere_isect.distance < closest_dist) {
            closest_dist = sphere_isect.distance;
            closest_hit = sphere_isect;
        }
    }

    // Check plane intersections
    for (int i = 0; i < d_num_planes; i++) {
        Intersection_t plane_isect = intersectPlane(ray, d_planes[i]);
        if (plane_isect.hit && plane_isect.distance < closest_dist) {
            closest_dist = plane_isect.distance;
            closest_hit = plane_isect;
        }
    }

    for (int i = 0; i < d_num_triangles; i++) {
        Intersection_t tri_isect = intersectTriangle(ray, d_triangles[i]);
        if (tri_isect.hit && tri_isect.distance < closest_dist) {
            closest_dist = tri_isect.distance;
            closest_hit = tri_isect;
        }
    }

    return closest_hit;
}

__device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float length(const float3& a) {
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ Color_t traceRay(Ray_t ray, curandState* rand_state, int max_depth) {
    Color_t final_color(0.0f);
    Color_t throughput(1.0f);

    for (int depth = 0; depth < max_depth; depth++) {
        Intersection_t isect = intersectScene(ray);
        if (!isect.hit) break;  // Ray missed everything, add background color (black)

        // Get emitted light
        final_color += throughput * isect.material->emitted(ray, isect);

        // Handle scattering
        ScatterRecord_t srec;
        if (!isect.material->scatter(ray, isect, srec, rand_state)) {
            break;
        }

        if (srec.is_specular) {
            throughput *= srec.attenuation;
            ray = srec.scattered_ray;
            continue;
        }

        // Update ray for next iteration
        ray = srec.scattered_ray;

        // Update throughput
        Color_t brdf = srec.attenuation * isect.material->scatteringPdf(ray, isect, srec.scattered_ray);
        throughput *= brdf * (1.0f / srec.pdf);

        // Russian roulette termination
        if (depth > 3) {
            float p = fmaxf(throughput.r, fmaxf(throughput.g, throughput.b));
            if (curand_uniform(rand_state) > p) {
                break;
            }
            throughput *= 1.0f / p;
        }
    }

    return final_color;
}

__device__ Ray_t generateCameraRay(float u, float v) {
    // Get camera data from constant memory
    const float aspect_ratio = static_cast<float>(d_camera.width) / d_camera.height;
    const float viewport_height = 2.0f * tanf(d_camera.fov * 0.5f);
    const float viewport_width = aspect_ratio * viewport_height;

    // Calculate viewport vectors
    Vec3f_t origin = Vec3f_t::fromFloat3(d_camera.origin);
    Vec3f_t forward = Vec3f_t::fromFloat3(d_camera.forward);
    Vec3f_t right = Vec3f_t::fromFloat3(d_camera.right);
    Vec3f_t up = Vec3f_t::fromFloat3(d_camera.up);

    // Calculate the point on the viewport
    float x_offset = (2.0f * u - 1.0f) * viewport_width * 0.5f;
    float y_offset = (2.0f * v - 1.0f) * viewport_height * 0.5f;

    Vec3f_t ray_direction = forward + right * x_offset + up * y_offset;
    ray_direction = ray_direction.normalized();

    return Ray_t(origin, ray_direction, Ray_t::Type::PRIMARY);
}

__device__ void initRand(curandState* rand_state, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = idy * gridDim.x * blockDim.x + idx;
    curand_init(seed + offset, 0, 0, rand_state);
}

__global__ void renderKernel(float4* output, uint32_t width, uint32_t height, uint32_t frame_count) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int pixel_index = y * width + x;

    // Initialize random state
    curandState rand_state;
    initRand(&rand_state, frame_count * width * height + pixel_index);

    // Calculate UV coordinates with jittering
    const float u = (x + curand_uniform(&rand_state)) / static_cast<float>(width);
    const float v = (y + curand_uniform(&rand_state)) / static_cast<float>(height);

    // Generate camera ray
    Ray_t ray = generateCameraRay(u, v);

    // Trace ray and accumulate color
    Color_t pixel_color = traceRay(ray, &rand_state, 50);  // Max depth of 50

    // Accumulate samples if frame_count > 0
    if (frame_count > 0) {
        float4 prev_color = output[pixel_index];
        float t = 1.0f / (frame_count + 1);
        pixel_color = pixel_color * t + Color_t(prev_color.x, prev_color.y, prev_color.z) * (1.0f - t);
    }

    // Write final color
    output[pixel_index] = make_float4(pixel_color.r, pixel_color.g, pixel_color.b, 1.0f);
}

// initialization function
void initializeGPUData(const Camera_t& camera, const Scene_t* d_scene) {
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
    CUDA_CHECK(cudaMemcpyToSymbol(d_camera, &h_camera, sizeof(GPUCamera)));
}


// this function to handle kernel launch
extern "C" void launchRenderKernel(float4* output, uint32_t width, uint32_t height, uint32_t frame_count, dim3 grid, dim3 block) {
    renderKernel<<<grid, block>>>(output, width, height, frame_count);
}
