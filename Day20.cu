#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel: Apply Rotary Embedding to a single query/key pair
__device__ void apply_rotary_embedding(
    float* q,
    float* k,
    int head_dim,
    int position,
    float base = 10000.0f
) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(base, (float)(i) / head_dim);
        float theta = position * freq;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float q_real = q[i];
        float q_imag = q[i + 1];
        float k_real = k[i];
        float k_imag = k[i + 1];

        q[i] = q_real * cos_theta - q_imag * sin_theta;
        q[i + 1] = q_real * sin_theta + q_imag * cos_theta;

        k[i] = k_real * cos_theta - k_imag * sin_theta;
        k[i + 1] = k_real * sin_theta + k_imag * cos_theta;
    }
}

// Kernel launcher
__global__ void rope_kernel(
    float* queries,
    float* keys,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_elements = batch_size * seq_len * num_heads;
    if (idx >= total_elements) return;

    int batch_idx = idx / (seq_len * num_heads);
    int seq_idx = (idx / num_heads) % seq_len;
    int head_idx = idx % num_heads;

    int base_index = batch_idx * seq_len * num_heads * head_dim
                   + seq_idx * num_heads * head_dim
                   + head_idx * head_dim;

    apply_rotary_embedding(&queries[base_index], &keys[base_index], head_dim, seq_idx);
}

// Host wrapper
void apply_rope(float* d_queries, float* d_keys, int B, int T, int H, int D) {
    int total_threads = B * T * H;
    dim3 blockSize(256);
    dim3 gridSize((total_threads + blockSize.x - 1) / blockSize.x);

    rope_kernel<<<gridSize, blockSize>>>(d_queries, d_keys, B, T, H, D);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Main program
int main() {
    const int B = 1;  // Batch size
    const int T = 4;  // Sequence length
    const int H = 2;  // Number of heads
    const int D = 4;  // Head dimension (must be even)

    const int total_elements = B * T * H * D;

    // Allocate host memory
    float* h_queries = new float[total_elements];
    float* h_keys    = new float[total_elements];

    // Initialize with test values
    for (int i = 0; i < total_elements; ++i) {
        h_queries[i] = 0.01f * i;
        h_keys[i]    = 0.02f * i;
    }

    std::cout << "Before RoPE:\n";
    for (int i = 0; i < total_elements; ++i) {
        std::cout << "Q[" << i << "] = " << h_queries[i]
                  << ", K[" << i << "] = " << h_keys[i] << std::endl;
    }

    // Allocate device memory
    float *d_queries, *d_keys;
    CHECK_CUDA(cudaMalloc((void**)&d_queries, total_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_keys, total_elements * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_queries, h_queries, total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_keys, h_keys, total_elements * sizeof(float), cudaMemcpyHostToDevice));

    // Apply RoPE
    apply_rope(d_queries, d_keys, B, T, H, D);

    // Copy back results
    CHECK_CUDA(cudaMemcpy(h_queries, d_queries, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_keys, d_keys, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\nAfter RoPE:\n";
    for (int i = 0; i < total_elements; ++i) {
        std::cout << "Q[" << i << "] = " << h_queries[i]
                  << ", K[" << i << "] = " << h_keys[i] << std::endl;
    }

    // Cleanup
    delete[] h_queries;
    delete[] h_keys;
    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_keys));

    return 0;
}

