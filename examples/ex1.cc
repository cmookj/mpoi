#include "core/mpoi.h"

#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>

using namespace std::chrono;

template <typename T>
int
sgn (T val) {
    return (T (0) < val) - (val < T (0));
}

int
main (int argc, char* argv[]) {
    mpoi pc ("./examples/kernel.cl");
    pc.display_platform_info();

    std::size_t kernel_id = pc.create_kernel ("vec_calc");

    constexpr std::size_t size = 64'000'000;

    auto a = std::make_unique<float[]> (size);
    auto b = std::make_unique<float[]> (size);
    auto c = std::make_unique<float[]> (size);

    for (std::size_t i = 0; i != size; i++) {
        a[i] = i;
        b[i] = size - i;
    }

    const std::size_t  num_trials = 10;
    std::vector<int>   duration_parallel (num_trials);
    std::vector<int>   duration_serial (num_trials);
    std::vector<float> difference_parallel_serial (num_trials);

    for (std::size_t i = 0; i < num_trials; i++) {
        auto t0 = high_resolution_clock::now();

        size_t a_buffer =
            pc.create_buffer (mpoi::buffer_property::READ_ONLY, size * sizeof (float));
        size_t b_buffer =
            pc.create_buffer (mpoi::buffer_property::READ_ONLY, size * sizeof (float));
        size_t c_buffer =
            pc.create_buffer (mpoi::buffer_property::READ_WRITE, size * sizeof (float));

        pc.enqueue_write_buffer (a_buffer, size * sizeof (float), a.get());
        pc.enqueue_write_buffer (b_buffer, size * sizeof (float), b.get());

        pc.set_kernel_argument (kernel_id, 0, a_buffer);
        pc.set_kernel_argument (kernel_id, 1, b_buffer);
        pc.set_kernel_argument (kernel_id, 2, c_buffer);

        pc.enqueue_data_parallel_kernel (kernel_id, size, 100);

        pc.enqueue_read_buffer (c_buffer, size * sizeof (float), c.get());

        pc.release_buffer (a_buffer);
        pc.release_buffer (b_buffer);
        pc.release_buffer (c_buffer);

        auto t1              = high_resolution_clock::now();
        duration_parallel[i] = static_cast<int> (duration_cast<milliseconds> (t1 - t0).count());

        auto d = std::make_unique<float[]> (size);
        t0     = high_resolution_clock::now();
        for (size_t j = 0; j != size; j++) {
            const float term = sin (a[j]) * cos (b[j]);
            d[j]             = exp (term + sgn (term) * cos (a[j]) * sin (b[j]));
        }

        t1                 = high_resolution_clock::now();
        duration_serial[i] = static_cast<int> (duration_cast<milliseconds> (t1 - t0).count());

        float diff = 0.f;
        for (size_t j = 0; j != size; j++) {
            diff += pow (c[j] - d[j], 2);
        }
        diff /= float (size);
        difference_parallel_serial[i] = diff;
    }

    std::cout << std::format ("\n\n{0:=^80}\n", " S U M M A R Y ");

    float sum_ratio    = 0.f;
    int   sum_parallel = 0;
    int   sum_serial   = 0;
    std::cout << std::format (
        "{0:^20}{1:^20}{2:^20}{3:^20}\n",
        "Parallel (msec)",
        "Serial (msec)",
        "Ratio (P/S)",
        "Difference"
    );
    std::cout << std::format ("{0:-^80}\n", "");

    for (unsigned i = 0; i != num_trials; i++) {
        std::cout << std::format (
            "{0:^20}{1:^20}{2:^20}{3:^20}\n",
            duration_parallel[i],
            duration_serial[i],
            float (duration_parallel[i]) / float (duration_serial[i]),
            difference_parallel_serial[i]
        );
        sum_parallel += duration_parallel[i];
        sum_serial += duration_serial[i];
        sum_ratio += float (duration_parallel[i]) / float (duration_serial[i]);
    }
    std::cout << std::format ("{0:-^80}\n", "");
    std::cout << std::format (
        "{0:^20}{1:^20}{2:^20}\n",
        float (sum_parallel) / float (num_trials),
        float (sum_serial) / float (num_trials),
        sum_ratio / float (num_trials)
    );

    return 0;
}
