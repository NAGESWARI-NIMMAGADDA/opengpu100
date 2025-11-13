#include <stdio.h>

// Device function for atomicAdd on long long integers
__device__ long long atomicAddLongLong(long long *address, long long value) {
    unsigned long long *uaddr = (unsigned long long *)address;
    unsigned long long old = *uaddr, assumed;
    do {
        assumed = old;
        old = atomicCAS(uaddr, assumed, assumed + value);
    } while (assumed != old);
    return (long long)old;
}

// Kernel function to perform atomic addition
__global__ void atomicAddKernel(long long *device_result) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // Debug print: each thread's contribution
    printf("Thread %d adding %d to global sum.\n", thread_id, thread_id);

    atomicAddLongLong(device_result, thread_id);
}

int main() {
    long long *device_sum;
    long long host_sum = 0;

    cudaMalloc(&device_sum, sizeof(long long));
    cudaMemcpy(device_sum, &host_sum, sizeof(long long), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = 4;

    printf("Launching kernel with %d blocks of %d threads each.\n", blocks_per_grid, threads_per_block);

    // Kernel launch
    atomicAddKernel<<<blocks_per_grid, threads_per_block>>>(device_sum);
    cudaDeviceSynchronize();

    cudaMemcpy(&host_sum, device_sum, sizeof(long long), cudaMemcpyDeviceToHost);

    printf("Final accumulated value: %lld\n", host_sum); // Expected: sum of all thread IDs

    cudaFree(device_sum);
    return 0;
}
