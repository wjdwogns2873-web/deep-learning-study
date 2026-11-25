#include "preprocess.cuh"
#include <algorithm>

__global__ void preprocessBilinearKernel(
    const uint8_t* __restrict__ src, int srcW, int srcH, int srcStep, 
    float* __restrict__ dst, int dstW, int dstH
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dstW || dy >= dstH) return;

    // OpenCV cv::resize (INTER_LINEAR)와 동일한 픽셀 중심 좌표 매핑
    float scaleX = static_cast<float>(srcW) / dstW;
    float scaleY = static_cast<float>(srcH) / dstH;

    float sx = (dx + 0.5f) * scaleX - 0.5f;
    float sy = (dy + 0.5f) * scaleY - 0.5f;

    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx1 = sx - x0;
    float wy1 = sy - y0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    x0 = max(0, min(x0, srcW - 1));
    x1 = max(0, min(x1, srcW - 1));
    y0 = max(0, min(y0, srcH - 1));
    y1 = max(0, min(y1, srcH - 1));

    const uint8_t* p00 = src + y0 * srcStep + x0 * 3;
    const uint8_t* p10 = src + y0 * srcStep + x1 * 3;
    const uint8_t* p01 = src + y1 * srcStep + x0 * 3;
    const uint8_t* p11 = src + y1 * srcStep + x1 * 3;

    float w00 = wx0 * wy0;
    float w10 = wx1 * wy0;
    float w01 = wx0 * wy1;
    float w11 = wx1 * wy1;

    constexpr float kNorm = 1.0f / 255.0f;
    float b = (p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11) * kNorm;
    float g = (p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11) * kNorm;
    float r = (p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11) * kNorm;

    int channelSize = dstW * dstH;
    int outIdx = dy * dstW + dx;

    dst[0 * channelSize + outIdx] = r;
    dst[1 * channelSize + outIdx] = g;
    dst[2 * channelSize + outIdx] = b;
}

void launchPreprocessKernel(
    const uint8_t* d_src, int srcW, int srcH, int srcStep, 
    float* d_dst, int dstW, int dstH, 
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    preprocessBilinearKernel<<<grid, block, 0, stream>>>(
        d_src, srcW, srcH, srcStep, d_dst, dstW, dstH
    );
}