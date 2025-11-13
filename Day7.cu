
#include <stdio.h>
#include <cuda_runtime.h>
#define Mask_width 5
#define shared_size (16 + Mask_width - 1)
__constant__ float M[Mask_width][Mask_width];
__global__ void twod_convolution_kernel(const float* A, float* C, int n) {
    int threadx = threadIdx.x;
    int thready = threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadx;
    int j = blockIdx.y * blockDim.y + thready;
    __shared__ float S_A[shared_size][shared_size];
    if (i < n && j < n) {
        S_A[threadx + Mask_width/2][thready + Mask_width/2] = A[i * n + j];
    } else {
        S_A[threadx + Mask_width/2][thready + Mask_width/2] = 0.0f;
    }
    if (threadx < Mask_width / 2) {
        int left = i - Mask_width / 2;
        S_A[threadx][thready + Mask_width/2] = (left >= 0 && j < n) ? A[left * n + j] : 0.0f;
    }
    if (threadx < Mask_width / 2) {
        int right = i + blockDim.x;
        S_A[threadx + blockDim.x + Mask_width/2][thready + Mask_width/2] = (right < n && j < n) ? A[right * n + j] : 0.0f;
    }
    if (thready < Mask_width / 2) {
        int top = j - Mask_width / 2;
        S_A[threadx + Mask_width/2][thready] = (top >= 0 && i < n) ? A[i * n + top] : 0.0f;
    }
    if (thready < Mask_width / 2) {
        int bottom = j + blockDim.y;
        S_A[threadx + Mask_width/2][thready + blockDim.y + Mask_width/2] = (bottom < n && i < n) ? A[i * n + bottom] : 0.0f;
    }
    __syncthreads();
    if (i < n && j < n) {
        float result = 0.0f;
        for (int k = 0; k < Mask_width; k++) {
            for (int x = 0; x < Mask_width; x++) {
                result += S_A[threadx + k][thready + x] * M[k][x];
            }
        }
        C[i * n + j] = result;
    }
}
void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
int main() {
    const int n = 10;
    float* h_A = (float*)malloc(n * n * sizeof(float));
    float* h_C = (float*)malloc(n * n * sizeof(float));
    float h_M[Mask_width][Mask_width];
    for (int i = 0; i < n * n; i++) h_A[i] = 3.0f;
    for (int i = 0; i < Mask_width; i++)
        for (int j = 0; j < Mask_width; j++)
            h_M[i][j] = 5.0f;
    float *d_A, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Copy A");
    cudaMemcpyToSymbol(M, h_M, Mask_width * Mask_width * sizeof(float));
    checkCudaError("Copy M");
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1)/blockSize.x, (n + blockSize.y - 1)/blockSize.y);
    twod_convolution_kernel<<<gridSize, blockSize>>>(d_A, d_C, n);
    cudaDeviceSynchronize();
    checkCudaError("Kernel");
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Copy result");
    printf("Output Matrix C:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.1f ", h_C[i * n + j]);
        }
        printf("\n");
    }
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
    return 0;
}
