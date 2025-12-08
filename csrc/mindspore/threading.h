#pragma once
#include <iostream>
#include <thread>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cerrno>
#include <pthread.h>
#include <sched.h>

/**
 * @brief An RAII class to temporarily set thread affinity and restore it on destruction.
 *
 * Usage:
 * {
 * ThreadAffinityGuard guard(8); // Binds thread to CPU 8
 * // ... do work on CPU 8 ...
 * } // Original affinity is automatically restored here
 */
class ThreadAffinityGuard {
public:
    /**
     * @brief Saves the current thread affinity and sets a new affinity to a single CPU.
     * @param cpu_id The CPU core to bind the current thread to.
     */
    explicit ThreadAffinityGuard(int cpu_id) {
        // 1. Get and save the original affinity mask.
        if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &original_mask_) != 0) {
            // If we can't get the original mask, we can't safely restore it.
            throw std::runtime_error("Failed to get original thread affinity.");
        }
        
        // 2. Create and set the new, temporary mask.
        cpu_set_t new_mask;
        CPU_ZERO(&new_mask);
        CPU_SET(cpu_id, &new_mask);
        
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &new_mask) != 0) {
            // We failed to set the new mask. For safety, restore the original immediately.
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &original_mask_); // Best effort restore
            throw std::runtime_error(std::string("Failed to set thread affinity to CPU ") + std::to_string(cpu_id));
        }
    }

    /**
     * @brief Destructor that automatically restores the original thread affinity.
     */
    ~ThreadAffinityGuard() {
        // 3. Restore the original affinity mask.
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &original_mask_) != 0) {
            // It's generally a bad idea to throw from a destructor.
            // A warning is more appropriate here.
            std::cerr << "Warning: Failed to restore original thread affinity." << std::endl;
        }
    }

    // Delete copy and move constructors/assignments to prevent misuse.
    ThreadAffinityGuard(const ThreadAffinityGuard&) = delete;
    ThreadAffinityGuard& operator=(const ThreadAffinityGuard&) = delete;
    ThreadAffinityGuard(ThreadAffinityGuard&&) = delete;
    ThreadAffinityGuard& operator=(ThreadAffinityGuard&&) = delete;

private:
    cpu_set_t original_mask_; // Member variable to store the original mask
};