#include <stdio.h>
#include <cuda_runtime.h>

#define TOTAL_ELEMENTS 1024   // Number of elements
#define LEARNING_RATE  0.5f   // Larger learning rate

// Mirror Maps
#define EUCLIDEAN_MAP       0  // Standard gradient descent
#define NEG_ENTROPY_MAP     1  // Exponentiated gradient descent
#define LOG_BARRIER_MAP     2  // Positive orthant

// CUDA Error check function
void checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", message, cudaGetErrorString(result));
        exit(-1);
    }
}

__global__ void mirrorDescentKernel(float *dev_params, float *dev_grads, float step_size, int map_type, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float updated_val = dev_params[idx];

    // Debug: print before update (for first few elements only)
    if (idx < 5) {
        printf("[Thread %d] Before update: param = %f, grad = %f\n", idx, dev_params[idx], dev_grads[idx]);
    }

    switch (map_type) {
        case EUCLIDEAN_MAP:
            updated_val = dev_params[idx] - step_size * dev_grads[idx];
            break;

        case NEG_ENTROPY_MAP:
            updated_val = dev_params[idx] * expf(-step_size * dev_grads[idx]);
            break;

        case LOG_BARRIER_MAP:
            updated_val = dev_params[idx] / (1.0f + step_size * dev_grads[idx]);
            break;

        default:
            updated_val = dev_params[idx];
    }

    dev_params[idx] = updated_val;

    // Debug: print after update
    if (idx < 5) {
        printf("[Thread %d] After update: param = %f\n", idx, updated_val);
    }
}

int main() {
    float *host_params, *host_grads;
    float *dev_params, *dev_grads;
    int selected_map = NEG_ENTROPY_MAP; // Choose the method

    printf("=== Mirror Descent CUDA Debug ===\n");
    printf("Elements: %d, Learning rate: %.2f, Map Type: %d\n", TOTAL_ELEMENTS, LEARNING_RATE, selected_map);

    // Allocate host memory
    host_params = (float*)malloc(TOTAL_ELEMENTS * sizeof(float));
    host_grads = (float*)malloc(TOTAL_ELEMENTS * sizeof(float));

    // Allocate device memory
    checkCudaError(cudaMalloc(&dev_params, TOTAL_ELEMENTS * sizeof(float)), "Alloc dev_params");
    checkCudaError(cudaMalloc(&dev_grads, TOTAL_ELEMENTS * sizeof(float)), "Alloc dev_grads");

    // Initialize parameters and gradients
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        host_params[i] = 1.0f;
        host_grads[i] = 0.5f * i;
    }
    printf("[Host] Initialized parameters and gradients.\n");

    // Copy to GPU
    checkCudaError(cudaMemcpy(dev_params, host_params, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice), "Memcpy host_params -> dev_params");
    checkCudaError(cudaMemcpy(dev_grads, host_grads, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice), "Memcpy host_grads -> dev_grads");

    // Kernel execution
    int threadsPerBlock = 256;
    int blocksPerGrid = (TOTAL_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;
    printf("[Host] Launching kernel with %d blocks of %d threads...\n", blocksPerGrid, threadsPerBlock);

    mirrorDescentKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_params, dev_grads, LEARNING_RATE, selected_map, TOTAL_ELEMENTS);
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    // Copy results back
    checkCudaError(cudaMemcpy(host_params, dev_params, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy dev_params -> host_params");

    // Print first 10 updated values
    printf("\n[Host] First 10 updated parameter values:\n");
    for (int i = 0; i < 10; i++) {
        printf("param[%d] = %f\n", i, host_params[i]);
    }

    // Cleanup
    free(host_params);
    free(host_grads);
    cudaFree(dev_params);
    cudaFree(dev_grads);

    printf("[Host] Program completed successfully.\n");
    return 0;
}
