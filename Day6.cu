
#include <cuda_runtime.h>
#include <iostream>

#define WIDTH 1024
#define HEIGHT 1024

__global__ void transposeMatrix(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;
    size_t size = width * height * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i);
    }

    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input data to device");

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    checkCudaError("Kernel execution failed");

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output data to host");

    bool success = true;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (h_output[i * height + j] != h_input[j * width + i]) {
                success = false;
                break;
            }
        }
    }

    std::cout << (success ? "Matrix transposition succeeded!" : "Matrix transposition failed!") << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
