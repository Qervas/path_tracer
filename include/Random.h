#pragma once

#include <random>
#include <thread>

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
};

// Define thread_local static members
thread_local std::mt19937 Random::generator_;
thread_local std::uniform_real_distribution<double> Random::distribution_(0.0, 1.0); 