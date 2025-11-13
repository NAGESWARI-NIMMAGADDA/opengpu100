#include <stdio.h>
#include <cuda_runtime.h>

// Custom atomicAdd for long long integers
__device__ long long atomicAddLongLong(long long *address, long long value) {
    unsigned long long *uaddr = (unsigned long long *)address;
    unsigned long long oldValue = *uaddr, assumed;
    do {
        assumed = oldValue;
        oldValue = atomicCAS(uaddr, assumed, assumed + value);
    } while (assumed != oldValue);
    return (long long)oldValue;
}

// Kernel to add thread IDs to the sum
__global__ void sumThreadIndicesKernel(long long *deviceSum) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    // Debug print from each thread (might be a lot of output for large grids)
    if (threadId < 10) { // Limit output to first 10 threads
        printf("[Device] Thread %d adding itself to sum\n", threadId);
    }

    atomicAddLongLong(deviceSum, threadId);
}

int main() {
    long long *deviceSumPtr;
    long long hostSum = 0;

    // Allocate memory on GPU
    cudaMalloc(&deviceSumPtr, sizeof(long long));
    cudaMemcpy(deviceSumPtr, &hostSum, sizeof(long long), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = 4;

    printf("[Host] Launching kernel with %d blocks, %d threads per block\n",
           blocksPerGrid, threadsPerBlock);

    // Launch kernel
    sumThreadIndicesKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceSumPtr);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&hostSum, deviceSumPtr, sizeof(long long), cudaMemcpyDeviceToHost);

    printf("[Host] Final accumulated value: %lld\n", hostSum);
    printf("[Host] Expected value: %lld\n", 
           (long long)((threadsPerBlock * blocksPerGrid - 1) * (threadsPerBlock * blocksPerGrid) / 2));

    // Free GPU memory
    cudaFree(deviceSumPtr);
    return 0;
}
