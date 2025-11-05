#include "core/mpoi.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Image {
    int                  width    = 0;
    int                  height   = 0;
    int                  channels = 3;
    std::vector<uint8_t> pixels;  // RGB triplets (size = width * height * channels)
};

Image
read_ppm_binary (const std::string& filename) {
    std::ifstream file (filename, std::ios::binary);
    if (!file) throw std::runtime_error ("Cannot open file: " + filename);

    std::string magic;
    file >> magic;
    if (magic != "P6") throw std::runtime_error ("Unsupported PPM format (expected P6)");

    // Skip comments
    file >> std::ws;  // skip whitespace
    while (file.peek() == '#') {
        std::string comment;
        std::getline (file, comment);
    }

    int width = 0, height = 0, maxval = 0;
    file >> width >> height >> maxval;
    if (width <= 0 || height <= 0 || maxval != 255) throw std::runtime_error ("Invalid PPM header");

    char ch;
    file.get (ch);  // consume one whitespace (usually '\n') after maxval

    // Read pixel data
    std::vector<uint8_t> pixels (width * height * 3);
    file.read (reinterpret_cast<char*> (pixels.data()), pixels.size());
    if (!file) throw std::runtime_error ("Error reading pixel data");

    return Image{width, height, 3, std::move (pixels)};
}

void
write_ppm_binary (const std::string& filename, const Image& img) {
    std::ofstream file (filename, std::ios::binary);
    if (!file) throw std::runtime_error ("Cannot open file: " + filename);

    // Header
    file << "P6\n";
    file << std::to_string (img.width) << " " << std::to_string (img.height) << "\n" << "255\n";

    // Body
    file.write (reinterpret_cast<const char*> (img.pixels.data()), img.pixels.size());
}

inline uint8_t
rgb_to_gray (uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint8_t> (0.299 * r + 0.587 * g + 0.114 * b);
}

Image
rgb_to_gray (const Image& img) {
    std::vector<uint8_t> pixels (img.width * img.height);
    for (std::size_t i : std::views::iota (0, img.width * img.height)) {
        pixels[i] = rgb_to_gray (img.pixels[3 * i], img.pixels[3 * i + 1], img.pixels[3 * i + 2]);
    }

    return Image{img.width, img.height, 1, std::move (pixels)};
}

Image
gray_to_rgb (const Image& img) {
    std::vector<uint8_t> pixels (img.width * img.height * 3);
    for (std::size_t i : std::views::iota (0, img.width * img.height)) {
        pixels[3 * i] = pixels[3 * i + 1] = pixels[3 * i + 2] = img.pixels[i];
    }

    return Image{img.width, img.height, 3, std::move (pixels)};
}

//// Gaussian kernel
struct GaussianKernel {
    const int              denom = 65536;
    const std::vector<int> params{
        1,
        8,
        28,
        56,
        70,
        56,
        28,
        8,
        1,
    };

    int
    size () const {
        return params.size();
    }

    float
    operator() (const std::size_t i, const std::size_t j) {
        const int offset = std::floor (params.size() / 2);
        return float (params[i + offset] * params[j + offset]) / float (denom);
    }
};

float
pixel_convoluted (const Image& img, const int i, const int j) {
    GaussianKernel kernel;
    float          cv   = 0.f;
    int            half = std::floor (kernel.size() / 2);

    for (int jj : std::views::iota (-half, half + 1)) {
        int sj = std::clamp (jj + j, 0, img.height - 1);
        for (int ii : std::views::iota (-half, half + 1)) {
            int si = std::clamp (ii + i, 0, img.width - 1);
            cv += kernel (ii, jj) * img.pixels[sj * img.width + si];
        }
    }

    return cv;
}

Image
image_convoluted (const Image& img) {
    std::vector<uint8_t> pixels (img.width * img.height);

    for (int j : std::views::iota (0, img.height)) {
        for (int i : std::views::iota (0, img.width)) {
            pixels[j * img.width + i] = pixel_convoluted (img, i, j);
        }
    }
    return Image{img.width, img.height, 1, std::move (pixels)};
}

int
convolution_serial (const std::string& filename, const bool save_gray = false) {
    Image src  = read_ppm_binary ("examples/lenna.ppm");
    Image gray = rgb_to_gray (src);

    if (save_gray) {
        Image gray_3ch = gray_to_rgb (gray);
        write_ppm_binary (filename + "_gray.ppm", gray_3ch);
    }

    // Convolution
    auto  t0         = std::chrono::high_resolution_clock::now();
    Image convoluted = image_convoluted (gray);
    auto  t1         = std::chrono::high_resolution_clock::now();
    auto  time_elapsed_msec =
        static_cast<int> (duration_cast<std::chrono::milliseconds> (t1 - t0).count());

    Image rgb = gray_to_rgb (convoluted);
    write_ppm_binary (filename + "_gray_conv_s.ppm", rgb);

    return time_elapsed_msec;
}

Image
image_convoluted_parallel (const Image& img, mpoi& pc, const std::size_t kernel_id) {
    const int size   = img.width * img.height;
    const int width  = img.width;
    const int height = img.height;

    std::vector<uint8_t> pixels (img.width * img.height);

    std::size_t in_buffer =
        pc.create_buffer (mpoi::buffer_property::READ_ONLY, size * sizeof (uint8_t));
    std::size_t out_buffer =
        pc.create_buffer (mpoi::buffer_property::READ_WRITE, size * sizeof (uint8_t));

    pc.enqueue_write_buffer (in_buffer, size * sizeof (uint8_t), img.pixels.data());

    pc.set_kernel_argument (kernel_id, 0, in_buffer);
    pc.set_kernel_argument (kernel_id, 1, out_buffer);
    pc.set_kernel_argument (kernel_id, 2, sizeof (int), &width);
    pc.set_kernel_argument (kernel_id, 3, sizeof (int), &height);

    pc.enqueue_data_parallel_kernel (kernel_id, 200, width, height);

    pc.enqueue_read_buffer (out_buffer, size * sizeof (uint8_t), pixels.data());

    pc.release_buffer (in_buffer);
    pc.release_buffer (out_buffer);

    return Image{img.width, img.height, 1, std::move (pixels)};
}

int
convolution_parallel (const std::string& filename) {
    Image src  = read_ppm_binary ("examples/lenna.ppm");
    Image gray = rgb_to_gray (src);

    mpoi pc ("./examples/kernel2.cl");
    pc.display_platform_info();

    std::size_t kernel_id = pc.create_kernel ("gaussian_blur");

    // Convolution
    auto  t0         = std::chrono::high_resolution_clock::now();
    Image convoluted = image_convoluted_parallel (gray, pc, kernel_id);
    auto  t1         = std::chrono::high_resolution_clock::now();
    auto  time_elapsed_msec =
        static_cast<int> (duration_cast<std::chrono::milliseconds> (t1 - t0).count());

    Image rgb = gray_to_rgb (convoluted);
    write_ppm_binary (filename + "_gray_conv_p.ppm", rgb);

    return time_elapsed_msec;
}

int
main (int argc, char* argv[]) {
    const int time_serial = convolution_serial ("examples/lenna.ppm", true);
    std::cout << std::format ("Running time for serial computation = {} msec\n", time_serial);

    const int time_parallel = convolution_parallel ("examples/lenna.ppm");
    std::cout << std::format ("Running time for parallel computation = {} msec\n", time_parallel);
}
