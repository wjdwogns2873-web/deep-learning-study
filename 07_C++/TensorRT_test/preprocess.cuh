#ifndef PREPROCESS_CUH
#define PREPROCESS_CUH

#include <cuda_runtime.h>
#include <cstdint>

void launchPreprocessKernel(
    const uint8_t* d_src, int srcW, int srcH, int srcStep, 
    float* d_dst, int dstW, int dstH, 
    cudaStream_t stream
);

#endif // PREPROCESS_CUH