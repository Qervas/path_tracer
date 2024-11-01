#pragma once

#include "Vec3.h"
#include <random>
#include <thread>
#include <chrono>

class Random {
private:
    // Thread-local random number generator
    static thread_local std::mt19937 generator_;
    static thread_local std::uniform_real_distribution<double> distribution_;

public:
    // Initialize the random number generator with a unique seed per thread
    static void init() {
        std::random_device rd;
        generator_.seed(rd() ^ (
            static_cast<unsigned long long>(std::hash<std::thread::id>{}(std::this_thread::get_id())) +
            static_cast<unsigned long long>(std::chrono::system_clock::now().time_since_epoch().count())
        ));
    }

    // Generate a random double in [0,1)
    [[nodiscard]] static double get() {
        return distribution_(generator_);
    }

    // Generate a random double in [min,max)
    [[nodiscard]] static double get(double min, double max) {
        return min + (max - min) * get();
    }

    // Generate a random point in unit disk
    [[nodiscard]] static Vec3d randomInUnitDisk() {
        while (true) {
            Vec3d p(
                2.0 * get() - 1.0,  // Range [-1,1]
                2.0 * get() - 1.0,  // Range [-1,1]
                0.0                 // Disk is in XY plane
            );
            if (p.length_squared() < 1.0) {
                return p;
            }
        }
    }
};

// Define thread_local static members
thread_local std::mt19937 Random::generator_;
thread_local std::uniform_real_distribution<double> Random::distribution_(0.0, 1.0);