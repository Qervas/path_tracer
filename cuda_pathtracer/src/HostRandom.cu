#include "Random.cuh"

// Define static members
std::mt19937 HostRandom_t::generator;
std::uniform_real_distribution<float> HostRandom_t::distribution(0.0f, 1.0f);

void HostRandom_t::init() {
    std::random_device rd;
    generator.seed(rd());
}

float HostRandom_t::get() {
    return distribution(generator);
}

float HostRandom_t::get(float min, float max) {
    return min + (max - min) * get();
}

Vec3f_t HostRandom_t::randomInUnitDisk() {
    while (true) {
        Vec3f_t p(2.0f * get() - 1.0f, 2.0f * get() - 1.0f, 0.0f);
        if (p.length_squared() < 1.0f) return p;
    }
} 