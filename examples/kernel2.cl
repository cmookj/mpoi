__constant float gaussian[9] = { 1, 8, 28, 56, 70, 56, 28, 8, 1 };

__kernel void gaussian_blur (
    __global const uchar* input,  // input image (grayscale, 0..255)
    __global uchar* output,       // output image
    const int w,                  // image width
    const int h                   // image height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= w || y >= h)
        return;

    float sum = 0.0f;
    float norm = 65536.0f; // 256^2 (sum of outer product weights)

    // 9x9 convolution centered on (x, y)
    for (int ky = -4; ky <= 4; ++ky) {
        int sy = clamp(y + ky, 0, h - 1);
        float wy = gaussian[ky + 4];
        for (int kx = -4; kx <= 4; ++kx) {
            int sx = clamp(x + kx, 0, w - 1);
            float wx = gaussian[kx + 4];
            float weight = wx * wy;
            sum += (float)input[sy * w + sx] * weight;
        }
    }

    output[y * w + x] = (uchar)(sum / norm);
}
