#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Parameters
#define NUM_CLUSTERS 2     
#define NUM_POINTS 1024     
#define THREADS_PER_BLOCK 256

// CUDA error check macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// E-step kernel
__global__ void eStepKernel(float* dev_points, int num_points, float* dev_means,
                            float* dev_stddev, float* dev_mixing_coeff,
                            float* dev_responsibilities) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_points) {
        float x = dev_points[idx];
        float probs[NUM_CLUSTERS];
        float sum_prob = 0.0f;
        
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            float diff = x - dev_means[k];
            float exponent = -0.5f * (diff * diff) / (dev_stddev[k] * dev_stddev[k]);
            float gauss = (1.0f / (sqrtf(2.0f * M_PI) * dev_stddev[k])) * expf(exponent);
            probs[k] = dev_mixing_coeff[k] * gauss;
            sum_prob += probs[k];
        }
        
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            dev_responsibilities[idx * NUM_CLUSTERS + k] = probs[k] / sum_prob;
        }
    }
}

// M-step accumulation kernel
__global__ void mStepKernel(float* dev_points, int num_points, float* dev_responsibilities,
                            float* dev_sum_gamma, float* dev_sum_x, float* dev_sum_x2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_points) {
        float x = dev_points[idx];
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            float gamma = dev_responsibilities[idx * NUM_CLUSTERS + k];
            atomicAdd(&dev_sum_gamma[k], gamma);
            atomicAdd(&dev_sum_x[k], gamma * x);
            atomicAdd(&dev_sum_x2[k], gamma * x * x);
        }
    }
}

int main() {
    srand(static_cast<unsigned>(time(NULL)));

    // Generate synthetic data
    float host_points[NUM_POINTS];
    for (int i = 0; i < NUM_POINTS; i++) {
        if (i < NUM_POINTS / 2)
            host_points[i] = 2.0f + static_cast<float>(rand()) / RAND_MAX;
        else
            host_points[i] = 8.0f + static_cast<float>(rand()) / RAND_MAX;
    }

    // Initial parameters (host)
    float host_means[NUM_CLUSTERS] = {1.0f, 9.0f};
    float host_stddev[NUM_CLUSTERS] = {1.0f, 1.0f};
    float host_mixing_coeff[NUM_CLUSTERS] = {0.5f, 0.5f};

    // Device arrays
    float *dev_points, *dev_means, *dev_stddev, *dev_mixing_coeff;
    float *dev_responsibilities, *dev_sum_gamma, *dev_sum_x, *dev_sum_x2;

    CUDA_CHECK(cudaMalloc(&dev_points, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_means, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_stddev, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_mixing_coeff, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_responsibilities, NUM_POINTS * NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_sum_gamma, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_sum_x, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_sum_x2, NUM_CLUSTERS * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(dev_points, host_points, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_means, host_means, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_stddev, host_stddev, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_mixing_coeff, host_mixing_coeff, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

    int num_blocks = (NUM_POINTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    float host_sum_gamma[NUM_CLUSTERS];
    float host_sum_x[NUM_CLUSTERS];
    float host_sum_x2[NUM_CLUSTERS];

    int max_iterations = 100;
    for (int iter = 0; iter < max_iterations; iter++) {
        // E-step
        eStepKernel<<<num_blocks, THREADS_PER_BLOCK>>>(dev_points, NUM_POINTS, dev_means, dev_stddev,
                                                       dev_mixing_coeff, dev_responsibilities);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reset accumulators
        CUDA_CHECK(cudaMemset(dev_sum_gamma, 0, NUM_CLUSTERS * sizeof(float)));
        CUDA_CHECK(cudaMemset(dev_sum_x, 0, NUM_CLUSTERS * sizeof(float)));
        CUDA_CHECK(cudaMemset(dev_sum_x2, 0, NUM_CLUSTERS * sizeof(float)));

        // M-step
        mStepKernel<<<num_blocks, THREADS_PER_BLOCK>>>(dev_points, NUM_POINTS, dev_responsibilities,
                                                       dev_sum_gamma, dev_sum_x, dev_sum_x2);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back sums
        CUDA_CHECK(cudaMemcpy(host_sum_gamma, dev_sum_gamma, NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_sum_x, dev_sum_x, NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_sum_x2, dev_sum_x2, NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));

        // Update parameters
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            if (host_sum_gamma[k] > 1e-6f) {
                host_means[k] = host_sum_x[k] / host_sum_gamma[k];
                float variance = host_sum_x2[k] / host_sum_gamma[k] - host_means[k] * host_means[k];
                host_stddev[k] = sqrtf(fmax(variance, 1e-6f));
                host_mixing_coeff[k] = host_sum_gamma[k] / NUM_POINTS;
            }
        }

        CUDA_CHECK(cudaMemcpy(dev_means, host_means, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_stddev, host_stddev, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_mixing_coeff, host_mixing_coeff, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

        // Debug printing
        if (iter % 10 == 0) {
            std::cout << "Iteration " << iter << ":\n";
            for (int k = 0; k < NUM_CLUSTERS; k++) {
                std::cout << "  Cluster " << k
                          << " -> mean: " << host_means[k]
                          << ", stddev: " << host_stddev[k]
                          << ", pi: " << host_mixing_coeff[k]
                          << ", sum_gamma: " << host_sum_gamma[k]
                          << "\n";
            }

            // Print sample responsibilities for first 5 points
            float sample_responsibilities[5 * NUM_CLUSTERS];
            CUDA_CHECK(cudaMemcpy(sample_responsibilities, dev_responsibilities,
                                  5 * NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "  Sample responsibilities:\n";
            for (int i = 0; i < 5; i++) {
                std::cout << "    Point " << i << ": ";
                for (int k = 0; k < NUM_CLUSTERS; k++) {
                    std::cout << sample_responsibilities[i * NUM_CLUSTERS + k] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    // Free GPU memory
    cudaFree(dev_points);
    cudaFree(dev_means);
    cudaFree(dev_stddev);
    cudaFree(dev_mixing_coeff);
    cudaFree(dev_responsibilities);
    cudaFree(dev_sum_gamma);
    cudaFree(dev_sum_x);
    cudaFree(dev_sum_x2);

    return 0;
}
