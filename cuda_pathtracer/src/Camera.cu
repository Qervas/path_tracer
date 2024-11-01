#include "Camera.cuh"

__host__ void Camera_t::initialize() {
    // Calculate camera basis vectors
    w_ = (origin_ - target_).normalized();  // Forward
    u_ = cross(up_, w_).normalized();       // Right
    v_ = cross(w_, u_);                     // Up

    // Calculate viewport dimensions
    float theta = settings_.fov * M_PI / 180.0f;
    float viewport_height = 2.0f * tanf(theta/2.0f);
    float viewport_width = viewport_height * settings_.aspect_ratio;

    // Calculate virtual screen vectors
    horizontal_ = u_ * viewport_width * settings_.focus_dist;
    vertical_ = v_ * viewport_height * settings_.focus_dist;

    // Calculate top-left corner of virtual screen
    top_left_ = origin_ - horizontal_/2.0f + vertical_/2.0f - w_ * settings_.focus_dist;

    // Calculate lens radius for depth of field
    lens_radius_ = settings_.aperture / 2.0f;
}

__device__ Point3f_t Camera_t::randomInUnitDisk(curandState* rand_state) const {
    float r = sqrtf(Random_t::get(rand_state));
    float theta = 2.0f * M_PI * Random_t::get(rand_state);
    return Point3f_t(r * cosf(theta), r * sinf(theta), 0.0f);
}

__device__ Ray_t Camera_t::getRayDevice(float s, float t, curandState* rand_state, bool jitter) const {
    Point3f_t ray_origin = origin_;
    
    if (settings_.use_dof) {
        Point3f_t rd = randomInUnitDisk(rand_state) * lens_radius_;
        Point3f_t offset = u_ * rd.x + v_ * rd.y;
        ray_origin = origin_ + offset;
    }

    if (jitter) {
        s += Random_t::get(rand_state) - 0.5f;
        t += Random_t::get(rand_state) - 0.5f;
    }

    Point3f_t screen_point = top_left_ + horizontal_ * s - vertical_ * t;
    return Ray_t(ray_origin, (screen_point - ray_origin).normalized());
}

__host__ Ray_t Camera_t::getRayHost(float s, float t, bool jitter) const {
    Point3f_t ray_origin = origin_;
    
    if (settings_.use_dof) {
        Point3f_t rd = HostRandom_t::randomInUnitDisk() * lens_radius_;
        Point3f_t offset = u_ * rd.x + v_ * rd.y;
        ray_origin = origin_ + offset;
    }

    if (jitter) {
        s += HostRandom_t::get() - 0.5f;
        t += HostRandom_t::get() - 0.5f;
    }

    Point3f_t screen_point = top_left_ + horizontal_ * s - vertical_ * t;
    return Ray_t(ray_origin, (screen_point - ray_origin).normalized());
}

__host__ __device__ Ray_t Camera_t::getRay(float s, float t, curandState* rand_state, bool jitter) const {
    #ifdef __CUDA_ARCH__
        return getRayDevice(s, t, rand_state, jitter);
    #else
        return getRayHost(s, t, jitter);
    #endif
}

__host__ __device__ Ray_t Camera_t::getRayForPixel(uint32_t x, uint32_t y, curandState* rand_state, bool jitter) const {
    float u = (static_cast<float>(x) + (jitter ? 
        #ifdef __CUDA_ARCH__
            Random_t::get(rand_state)
        #else
            HostRandom_t::get()
        #endif
        : 0.5f)) / (settings_.width - 1);
    
    float v = 1.0f - (static_cast<float>(y) + (jitter ? 
        #ifdef __CUDA_ARCH__
            Random_t::get(rand_state)
        #else
            HostRandom_t::get()
        #endif
        : 0.5f)) / (settings_.height - 1);

    return getRay(u, v, rand_state, false);
}

// Camera movement methods
__host__ void Camera_t::lookAt(const Point3f_t& target) {
    target_ = target;
    initialize();
}

__host__ void Camera_t::lookAt(const Point3f_t& position, const Point3f_t& target, const Vec3f_t& up) {
    origin_ = position;
    target_ = target;
    up_ = up.normalized();
    initialize();
}

__host__ void Camera_t::setPosition(const Point3f_t& position) {
    origin_ = position;
    initialize();
}

__host__ void Camera_t::moveForward(float distance) {
    origin_ -= w_ * distance;
    initialize();
}

__host__ void Camera_t::moveRight(float distance) {
    origin_ += u_ * distance;
    initialize();
}

__host__ void Camera_t::moveUp(float distance) {
    origin_ += v_ * distance;
    initialize();
}

__host__ void Camera_t::rotateYaw(float angle) {
    yaw_ = angle;
    updateRotation();
}

__host__ void Camera_t::rotatePitch(float angle) {
    pitch_ = angle;
    updateRotation();
}

__host__ void Camera_t::updateRotation() {
    Vec3f_t forward(
        cosf(pitch_) * cosf(yaw_),
        sinf(pitch_),
        cosf(pitch_) * sinf(yaw_)
    );
    
    target_ = origin_ + forward;
    initialize();
} 