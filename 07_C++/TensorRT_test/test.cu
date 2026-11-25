#include <iostream>
#include <cuda_runtime.h>

// GPU에서 실행될 커널 함수
__global__ void multiplyByTwoKernel(const float* d_in, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

int main() {
    const int N = 5;
    const size_t bytes = N * sizeof(float);

    // Host(CPU) 메모리 준비 및 값 채우기
    float h_in[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_out[N] = {0.0f};

    // Device(GPU) 메모리 할당
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // CPU -> GPU 데이터 복사
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    multiplyByTwoKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    std::cout << "결과: ";
    for (int i = 0; i < N; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}