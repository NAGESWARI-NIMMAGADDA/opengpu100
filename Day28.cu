#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cuComplex.h>

#define NUM_POINTS 1024      // Number of spatial points
#define DX 0.01              // Spatial step
#define DT 5e-7              // Time step (reduced for stability)
#define HBAR 1.0
#define MASS 1.0
#define BLOCK_SIZE 256

using complexd = double2;

// --- Complex number helpers ---
__host__ __device__ inline complexd make_cplx(double r, double i) {
    return make_double2(r, i);
}
__host__ __device__ inline complexd cplx_add(complexd a, complexd b) {
    return make_double2(a.x + b.x, a.y + b.y);
}
__host__ __device__ inline complexd cplx_sub(complexd a, complexd b) {
    return make_double2(a.x - b.x, a.y - b.y);
}
__host__ __device__ inline complexd cplx_mul_scalar(double s, complexd a) {
    return make_double2(s * a.x, s * a.y);
}
__host__ __device__ inline complexd cplx_mul(complexd a, complexd b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__host__ __device__ inline double cplx_norm(complexd a) {
    return a.x * a.x + a.y * a.y;
}

// --- Kernel: evolve wavefunction ---
__global__ void evolve_wavefunction(complexd *curr_wave, complexd *next_wave, double *potentials) {
    __shared__ complexd shared_wave[BLOCK_SIZE + 2];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x + 1;

    if (idx < NUM_POINTS) {
        shared_wave[local_idx] = curr_wave[idx];
    }
    if (threadIdx.x == 0) {
        shared_wave[0] = (idx > 0) ? curr_wave[idx - 1] : curr_wave[idx];
    }
    if (threadIdx.x == blockDim.x - 1) {
        shared_wave[local_idx + 1] = (idx < NUM_POINTS - 1) ? curr_wave[idx + 1] : curr_wave[idx];
    }
    __syncthreads();

    if (idx > 0 && idx < NUM_POINTS - 1) {
        complexd laplacian = cplx_sub(cplx_add(shared_wave[local_idx - 1], shared_wave[local_idx + 1]),
                                      cplx_mul_scalar(2.0, shared_wave[local_idx]));
        laplacian = cplx_mul_scalar(1.0 / (DX * DX), laplacian);

        complexd i_hbar = make_cplx(0.0, HBAR);
        complexd term1 = cplx_mul_scalar(DT / (2.0 * MASS), cplx_mul(i_hbar, laplacian));
        complexd term2a = cplx_mul_scalar(potentials[idx], curr_wave[idx]);
        complexd term2 = cplx_mul_scalar(DT / HBAR, cplx_mul(i_hbar, term2a));

        next_wave[idx] = cplx_add(cplx_sub(curr_wave[idx], term1), term2);
    }
}

// --- Kernel: partial normalization ---
__global__ void compute_block_norms(complexd *wave, double *block_sums) {
    __shared__ double local_norms[BLOCK_SIZE];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    local_norms[tid] = (idx < NUM_POINTS) ? cplx_norm(wave[idx]) : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            local_norms[tid] += local_norms[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = local_norms[0];
    }
}

// --- Kernel: sum block norms ---
__global__ void sum_block_results(double *block_sums, int num_blocks, double *total_norm) {
    __shared__ double shared_sums[BLOCK_SIZE];
    int tid = threadIdx.x;

    shared_sums[tid] = (tid < num_blocks) ? block_sums[tid] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *total_norm = shared_sums[0];
    }
}

// --- Kernel: apply normalization ---
__global__ void apply_normalization(complexd *wave, double total_norm) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_POINTS && total_norm > 0.0) {
        double scale = rsqrt(total_norm);
        wave[idx] = cplx_mul_scalar(scale, wave[idx]);
    }
}

int main() {
    complexd *dev_wave, *dev_wave_next;
    double *dev_potentials, *dev_norm_factor, *dev_block_sums;
    complexd host_wave[NUM_POINTS];
    double host_potentials[NUM_POINTS], host_norm_factor = 0.0;

    int num_blocks = (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "[INIT] Generating initial wavefunction and potential...\n";

    // --- Initialize wavefunction & potential ---
    for (int i = 0; i < NUM_POINTS; i++) {
        double x = (i - NUM_POINTS / 2) * DX;
        double envelope = exp(-x * x);
        host_wave[i] = make_cplx(envelope * cos(5.0 * x), envelope * sin(5.0 * x));
        host_potentials[i] = 0.5 * x * x;
    }

    // --- Allocate GPU memory ---
    cudaMalloc(&dev_wave, NUM_POINTS * sizeof(complexd));
    cudaMalloc(&dev_wave_next, NUM_POINTS * sizeof(complexd));
    cudaMalloc(&dev_potentials, NUM_POINTS * sizeof(double));
    cudaMalloc(&dev_norm_factor, sizeof(double));
    cudaMalloc(&dev_block_sums, num_blocks * sizeof(double));

    cudaMemcpy(dev_wave, host_wave, NUM_POINTS * sizeof(complexd), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_potentials, host_potentials, NUM_POINTS * sizeof(double), cudaMemcpyHostToDevice);

    int threads_per_block = BLOCK_SIZE;

    std::cout << "[SIM] Starting time evolution...\n";

    // --- Time evolution ---
    for (int step = 0; step < 1000; step++) {
        evolve_wavefunction<<<num_blocks, threads_per_block>>>(dev_wave, dev_wave_next, dev_potentials);
        cudaDeviceSynchronize();

        std::swap(dev_wave, dev_wave_next);

        if (step % 100 == 0) {
            std::cout << "  -> Step " << step << ": Normalizing wavefunction...\n";
            compute_block_norms<<<num_blocks, threads_per_block>>>(dev_wave, dev_block_sums);
            cudaDeviceSynchronize();

            cudaMemset(dev_norm_factor, 0, sizeof(double));
            sum_block_results<<<1, threads_per_block>>>(dev_block_sums, num_blocks, dev_norm_factor);
            cudaDeviceSynchronize();

            cudaMemcpy(&host_norm_factor, dev_norm_factor, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << "     Norm before normalization: " << host_norm_factor << "\n";

            apply_normalization<<<num_blocks, threads_per_block>>>(dev_wave, host_norm_factor);
            cudaDeviceSynchronize();
        }
    }

    // --- Copy result back ---
    cudaMemcpy(host_wave, dev_wave, NUM_POINTS * sizeof(complexd), cudaMemcpyDeviceToHost);

    cudaFree(dev_wave);
    cudaFree(dev_wave_next);
    cudaFree(dev_potentials);
    cudaFree(dev_norm_factor);
    cudaFree(dev_block_sums);

    std::cout << "[RESULT] Sampled wavefunction values:\n";
    for (int i = 0; i < NUM_POINTS; i += NUM_POINTS / 10) {
        std::cout << "x = " << (i - NUM_POINTS / 2) * DX
                  << " | Psi = (" << host_wave[i].x << ", " << host_wave[i].y << ")\n";
    }

    return 0;
}
