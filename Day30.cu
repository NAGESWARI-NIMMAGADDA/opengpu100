#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define RADIUS 1
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * RADIUS)
#define FILTER_SIZE (2 * RADIUS + 1)

__constant__ float filter[FILTER_SIZE][FILTER_SIZE];

__global__ void conv2D(float *input, float *output, int N, int M) 
{
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - RADIUS;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - RADIUS;

    __shared__ float N_shared[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < N && col >= 0 && col < M)
        N_shared[threadIdx.y][threadIdx.x] = input[row * M + col];
    else
        N_shared[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    int tile_row = threadIdx.y - RADIUS;
    int tile_col = threadIdx.x - RADIUS;

    if (tile_row >= 0 && tile_row < OUT_TILE_DIM && tile_col >= 0 && tile_col < OUT_TILE_DIM) 
    {
        int out_row = blockIdx.y * OUT_TILE_DIM + tile_row;
        int out_col = blockIdx.x * OUT_TILE_DIM + tile_col;
        if (out_row < N && out_col < M) 
        {
            float val = 0.0f;
            for (int row_f = 0; row_f < FILTER_SIZE; row_f++) 
            {
                for (int col_f = 0; col_f < FILTER_SIZE; col_f++) 
                {
                    val += filter[row_f][col_f] * N_shared[threadIdx.y + row_f - RADIUS][threadIdx.x + col_f - RADIUS];
                }
            }
            output[out_row * M + out_col] = val;
        }
    }
}

void conv2D_cpu(float *input, float *output, float *host_filter, int N, int M) 
{
    for (int row = 0; row < N; row++) 
    {
        for (int col = 0; col < M; col++) 
        {
            float val = 0.0f;
            for (int row_f = 0; row_f < FILTER_SIZE; row_f++) 
            {
                for (int col_f = 0; col_f < FILTER_SIZE; col_f++) 
                {
                    int cur_row = row - RADIUS + row_f;
                    int cur_col = col - RADIUS + col_f;
                    if (cur_row >= 0 && cur_row < N && cur_col >= 0 && cur_col < M) 
                    {
                        val += host_filter[row_f * FILTER_SIZE + col_f] * input[cur_row * M + cur_col];
                    }
                }
            }
            output[row * M + col] = val;
        }
    }
}

void conv2D_cudnn(float *input_gpu, float *output_gpu, float *filter_gpu, int numRows, int numCols, float* gpu_time_cudnn) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, numRows, numCols);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, numRows, numCols);
    cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, FILTER_SIZE, FILTER_SIZE);
    cudnnSetConvolution2dDescriptor(convolution_descriptor, RADIUS, RADIUS, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    cudnnFindConvolutionForwardAlgorithm(cudnn, input_descriptor, filter_descriptor,convolution_descriptor,output_descriptor, 1, &returnedAlgoCount, &perfResults);
    
    cudnnConvolutionFwdAlgo_t convolution_algorithm = perfResults.algo;

    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_size);

    void *workspace_gpu = NULL;
    if (workspace_size > 0)
    {
        cudaMalloc(&workspace_gpu, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, input_gpu, filter_descriptor, filter_gpu, convolution_descriptor, convolution_algorithm, workspace_gpu, workspace_size, &beta, output_descriptor, output_gpu);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(gpu_time_cudnn, start, stop);

    if (workspace_gpu)
    {
        cudaFree(workspace_gpu);
    }

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);
}

__global__ void init(float *input, int N, int M, unsigned int seed) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) 
    {
        curandState state;
        curand_init(seed, row * M + col, 0, &state);
        input[row * M + col] = curand_uniform(&state);
    }
}

__global__ void heNormal_kernel(float *filter_temp, unsigned int seed) 
{
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int f_size = 2 * RADIUS + 1;
    int total = f_size * f_size;

    if (t_x < total) 
    {
        float fan_in = (float)total;
        float std_dev = sqrtf(2.0f / fan_in);
        
        curandState state;
        curand_init(seed, t_x, 0, &state);
        filter_temp[t_x] = curand_normal(&state) * std_dev;
    }
}

int main() 
{
    int N = 1<<10, M = 1<<11;
    float gpu_time_custom, gpu_time_cudnn, cpu_time;

    float *in_gpu, *out_gpu_custom, *out_gpu_cudnn, *cpu_output;
    cudaMallocManaged(&in_gpu, N * M * sizeof(float));
    cudaMallocManaged(&out_gpu_custom, N * M * sizeof(float));
    cudaMallocManaged(&out_gpu_cudnn, N * M * sizeof(float));
    cudaMallocManaged(&cpu_output, N * M * sizeof(float));

    dim3 threads(32, 32);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    init<<<blocks, threads>>>(in_gpu, N, M, time(NULL));
    cudaDeviceSynchronize();
    
    int f_size = FILTER_SIZE * FILTER_SIZE;
    float *managed_filter;
    cudaMallocManaged(&managed_filter, f_size * sizeof(float));
    heNormal_kernel<<<(f_size + 255) / 256, 256>>>(managed_filter, time(NULL));
    cudaDeviceSynchronize();

    cudaMemcpyToSymbol(filter, managed_filter, f_size * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    conv2D<<<blocks, threads>>>(in_gpu, out_gpu_custom, N, M);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_custom, start, stop);
    
    conv2D_cudnn(in_gpu, out_gpu_cudnn, managed_filter, N, M, &gpu_time_cudnn);

    clock_t cpu_start = clock();
    conv2D_cpu(in_gpu, cpu_output, managed_filter, N, M);
    cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    printf("GPU Convolution Execution Time: %.2f ms\n", gpu_time_custom);
    printf("cuDNN Convolution Execution Time: %.2f ms\n", gpu_time_cudnn);
    printf("CPU Convolution Execution Time: %.2f ms\n", cpu_time);

    cudaFree(managed_filter);
    cudaFree(in_gpu);
    cudaFree(out_gpu_custom);
    cudaFree(out_gpu_cudnn);
    cudaFree(cpu_output);

    return 0;
}
