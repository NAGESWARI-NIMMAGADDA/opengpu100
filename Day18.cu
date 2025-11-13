#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

int main() {
    // Matrix dimensions
    int rowsA = 2, colsB = 3, sharedDim = 4;

    // Host memory allocation
    float *hostMatrixA = (float *)malloc(rowsA * sharedDim * sizeof(float));
    float *hostMatrixB = (float *)malloc(sharedDim * colsB * sizeof(float));
    float *hostMatrixC = (float *)malloc(rowsA * colsB * sizeof(float));

    printf("Initializing matrices A and B on host...\n");

    // Initialize matrix A (rowsA x sharedDim)
    for (int i = 0; i < rowsA; i++)
        for (int j = 0; j < sharedDim; j++)
            hostMatrixA[i * sharedDim + j] = (float)(i + j);

    // Initialize matrix B (sharedDim x colsB)
    for (int i = 0; i < sharedDim; i++)
        for (int j = 0; j < colsB; j++)
            hostMatrixB[i * colsB + j] = (float)(i + j);

    // Device memory allocation
    float *devMatrixA, *devMatrixB, *devMatrixC;
    printf("Allocating matrices on device...\n");
    cudaMalloc(&devMatrixA, rowsA * sharedDim * sizeof(float));
    cudaMalloc(&devMatrixB, sharedDim * colsB * sizeof(float));
    cudaMalloc(&devMatrixC, rowsA * colsB * sizeof(float));

    // Copy matrices A and B to device
    printf("Copying data from host to device...\n");
    cudaMemcpy(devMatrixA, hostMatrixA, rowsA * sharedDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devMatrixB, hostMatrixB, sharedDim * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    printf("Creating cuBLAS handle...\n");
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // GEMM: C = alpha * A * B + beta * C
    const float alpha = 1.0f, beta = 0.0f;

    printf("Launching cuBLAS SGEMM for matrix multiplication...\n");
    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                rowsA, colsB, sharedDim,
                &alpha,
                devMatrixA, rowsA,
                devMatrixB, sharedDim,
                &beta,
                devMatrixC, rowsA);

    // Copy result matrix C from device to host
    printf("Copying result matrix C from device to host...\n");
    cudaMemcpy(hostMatrixC, devMatrixC, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Display matrices
    printf("\nMatrix A (%dx%d):\n", rowsA, sharedDim);
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < sharedDim; j++) {
            printf("%6.1f ", hostMatrixA[i * sharedDim + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B (%dx%d):\n", sharedDim, colsB);
    for (int i = 0; i < sharedDim; i++) {
        for (int j = 0; j < colsB; j++) {
            printf("%6.1f ", hostMatrixB[i * colsB + j]);
        }
        printf("\n");
    }

    printf("\nMatrix C = A * B (%dx%d):\n", rowsA, colsB);
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            printf("%6.1f ", hostMatrixC[i + j * rowsA]);  // Column-major layout
        }
        printf("\n");
    }

    // Clean up
    printf("\nCleaning up...\n");
    free(hostMatrixA);
    free(hostMatrixB);
    free(hostMatrixC);
    cudaFree(devMatrixA);
    cudaFree(devMatrixB);
    cudaFree(devMatrixC);
    cublasDestroy(cublasHandle);

    printf("Done.\n");
    return 0;
}
