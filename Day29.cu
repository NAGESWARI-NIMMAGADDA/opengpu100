#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) do {                                          \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        printf("CUDA Error at %s %d: %s\n", __FILE__, __LINE__,      \
               cudaGetErrorString(err));                              \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while(0)

// Changed constants
const int DATA_SIZE = 100000;        // Smaller data size
const int LOOP_COUNT = 10000;        // More iterations
const int THREADS_PER_BLOCK = 256;

__global__ void kernelAdd(float* arr1, float* arr2, float* arrOut, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arrOut[idx] = arr1[idx] + arr2[idx];
    }
}

__global__ void kernelScale(float* arr, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] *= scalar;
    }
}

__global__ void kernelSquare(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = arr[idx] * arr[idx];
    }
}

__global__ void kernelOffset(float* arr, float offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] += offset;
    }
}

void printTiming(const char* label, float ms) {
    printf("%s: %.3f ms\n", label, ms);
}

void verifyOutput(float* h_in1, float* h_in2, float* h_out, float* h_expected, int n) {
    for (int i = 0; i < n; i++) {
        float temp = h_in1[i] + h_in2[i];
        temp *= 2.0f;
        temp *= temp;
        h_expected[i] = temp + 1.0f;
    }

    bool match = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_expected[i] - h_out[i]) > 1e-5) {
            match = false;
            printf("Mismatch at index %d: Expected %f, Got %f\n", 
                   i, h_expected[i], h_out[i]);
            break;
        }
    }
    if (match) {
        printf("Verification successful! All results match.\n");
    }
}

int main() {
    float *h_in1, *h_in2, *h_out, *h_expected;
    float *d_in1, *d_in2, *d_out;
    size_t size = DATA_SIZE * sizeof(float);

    printf("Allocating host memory...\n");
    h_in1 = (float*)malloc(size);
    h_in2 = (float*)malloc(size);
    h_out = (float*)malloc(size);
    h_expected = (float*)malloc(size);

    printf("Initializing host arrays...\n");
    for (int i = 0; i < DATA_SIZE; i++) {
        h_in1[i] = rand() / (float)RAND_MAX;
        h_in2[i] = rand() / (float)RAND_MAX;
    }

    printf("Allocating device memory...\n");
    CHECK_CUDA(cudaMalloc(&d_in1, size));
    CHECK_CUDA(cudaMalloc(&d_in2, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    printf("Copying data to GPU...\n");
    CHECK_CUDA(cudaMemcpy(d_in1, h_in1, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in2, h_in2, size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blocks = (DATA_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Running warmup iterations...\n");
    for (int i = 0; i < 10; i++) {
        kernelAdd<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_in1, d_in2, d_out, DATA_SIZE);
        kernelScale<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, 2.0f, DATA_SIZE);
        kernelSquare<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, DATA_SIZE);
        kernelOffset<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, 1.0f, DATA_SIZE);
    }
    cudaStreamSynchronize(stream);

    printf("Running traditional loop execution...\n");
    cudaEventRecord(start, stream);
    for (int i = 0; i < LOOP_COUNT; i++) {
        kernelAdd<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_in1, d_in2, d_out, DATA_SIZE);
        kernelScale<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, 2.0f, DATA_SIZE);
        kernelSquare<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, DATA_SIZE);
        kernelOffset<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, 1.0f, DATA_SIZE);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_no_graph;
    cudaEventElapsedTime(&ms_no_graph, start, stop);
    printTiming("Without CUDA Graphs", ms_no_graph);

    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    printf("Capturing CUDA Graph...\n");
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    kernelAdd<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_in1, d_in2, d_out, DATA_SIZE);
    kernelScale<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, 2.0f, DATA_SIZE);
    kernelSquare<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, DATA_SIZE);
    kernelOffset<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, 1.0f, DATA_SIZE);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    printf("Executing with CUDA Graph...\n");
    cudaEventRecord(start, stream);
    for (int i = 0; i < LOOP_COUNT; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_graph;
    cudaEventElapsedTime(&ms_graph, start, stop);
    printTiming("With CUDA Graphs", ms_graph);

    printf("Verifying results...\n");
    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    verifyOutput(h_in1, h_in2, h_out, h_expected, DATA_SIZE);

    printf("Cleaning up...\n");
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
    free(h_in1);
    free(h_in2);
    free(h_out);
    free(h_expected);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);

    printf("Done!\n");
    return 0;
}
