
#include <iostream>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

#define SRAM_SIZE 1024
#define sequence_length 2
#define embed_dimension 2

constexpr int Block_column_size = SRAM_SIZE / (4 * embed_dimension);
constexpr int Block_row_size = std::min(SRAM_SIZE / (4 * embed_dimension), embed_dimension);

static_assert(Block_column_size > 0, "Block_column_size must be greater than 0");
static_assert(Block_row_size > 0, "Block_row_size must be greater than 0");

constexpr int Total_row_blocks = (sequence_length + Block_row_size - 1) / Block_row_size;
constexpr int Total_column_blocks = (sequence_length + Block_column_size - 1) / Block_column_size;

__global__ void flashAttentionForward(
    const float *Query,
    const float *Key,
    const float *Value,
    float *Output,
    float *max_values,
    float *sum_values,
    const float attention_scale)
{
    int thread_idx = threadIdx.x;
    float attention_scores[Block_row_size * Block_column_size];
    float attention_weights[Block_row_size * Block_column_size];
    float Query_block[Block_row_size * embed_dimension];
    float Key_block[Block_column_size * embed_dimension];
    float Value_block[Block_column_size * embed_dimension];

    for (int col_block = 0; col_block < Total_column_blocks; ++col_block)
    {
        if (thread_idx < Block_column_size) {
            for (int d = 0; d < embed_dimension; ++d) {
                Key_block[thread_idx * embed_dimension + d] = 
                    Key[col_block * Block_column_size * embed_dimension + thread_idx * embed_dimension + d];
                Value_block[thread_idx * embed_dimension + d] = 
                    Value[col_block * Block_column_size * embed_dimension + thread_idx * embed_dimension + d];
            }
        }
        __syncthreads();

        for (int row_block = 0; row_block < Total_row_blocks; ++row_block)
        {
            if (thread_idx < Block_row_size) {
                for (int d = 0; d < embed_dimension; ++d) {
                    Query_block[thread_idx * embed_dimension + d] = 
                        Query[row_block * Block_row_size * embed_dimension + thread_idx * embed_dimension + d];
                }
            }
            __syncthreads();

            if (thread_idx < Block_row_size) {
                float row_max = -1e20;
                for (int k = 0; k < Block_column_size; ++k) {
                    float score = 0.0f;
                    for (int d = 0; d < embed_dimension; ++d) {
                        score += Query_block[thread_idx * embed_dimension + d] * 
                                Key_block[k * embed_dimension + d];
                    }
                    score *= attention_scale;
                    attention_scores[thread_idx * Block_column_size + k] = score;
                    row_max = fmaxf(row_max, score);
                }
                float row_sum = 0.0f;
                for (int k = 0; k < Block_column_size; ++k) {
                    float weight = expf(attention_scores[thread_idx * Block_column_size + k] - row_max);
                    attention_weights[thread_idx * Block_column_size + k] = weight;
                    row_sum += weight;
                }
                for (int d = 0; d < embed_dimension; ++d) {
                    float weighted_sum = 0.0f;
                    for (int k = 0; k < Block_column_size; ++k) {
                        weighted_sum += attention_weights[thread_idx * Block_column_size + k] * 
                                      Value_block[k * embed_dimension + d];
                    }
                    Output[row_block * Block_row_size * embed_dimension + thread_idx * embed_dimension + d] = 
                        (row_sum > 0) ? (weighted_sum / row_sum) : 0;
                }
            }
            __syncthreads();
        }
    }
}

int main()
{
    float (*Query)[embed_dimension] = new float[sequence_length][embed_dimension];
    float (*Key)[embed_dimension] = new float[sequence_length][embed_dimension];
    float (*Value)[embed_dimension] = new float[sequence_length][embed_dimension];
    float (*Output)[embed_dimension] = new float[sequence_length][embed_dimension];
    float *sum_values = new float[sequence_length]();
    float *max_values = new float[sequence_length];

    for (int i = 0; i < sequence_length; i++) {
        max_values[i] = -1e20;
    }

    for (int i = 0; i < sequence_length; i++) {
        for (int j = 0; j < embed_dimension; j++) {
            Query[i][j] = 2.0f * rand() / RAND_MAX - 1.0f;
            Key[i][j] = 2.0f * rand() / RAND_MAX - 1.0f;
            Value[i][j] = 2.0f * rand() / RAND_MAX - 1.0f;
            Output[i][j] = 0.0f;
        }
    }

    float *device_Query, *device_Key, *device_Value, *device_Output;
    float *device_max_values, *device_sum_values;

    cudaMalloc(&device_Query, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&device_Key, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&device_Value, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&device_Output, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&device_sum_values, sequence_length * sizeof(float));
    cudaMalloc(&device_max_values, sequence_length * sizeof(float));

    cudaMemcpy(device_Query, Query, sequence_length * embed_dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_Key, Key, sequence_length * embed_dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_Value, Value, sequence_length * embed_dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_Output, Output, sequence_length * embed_dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_sum_values, sum_values, sequence_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_max_values, max_values, sequence_length * sizeof(float), cudaMemcpyHostToDevice);

    float attention_scale = 1.0f / sqrt(embed_dimension);

    dim3 block_dim(Block_row_size);
    dim3 grid_dim(1);

    flashAttentionForward<<<grid_dim, block_dim>>>(
        device_Query,
        device_Key,
        device_Value,
        device_Output,
        device_max_values,
        device_sum_values,
        attention_scale
    );

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaMemcpy(Output, device_Output, sequence_length * embed_dimension * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_values, device_max_values, sequence_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_values, device_sum_values, sequence_length * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Query:" << std::endl;
    for (int i = 0; i < sequence_length; i++) {
        for (int j = 0; j < embed_dimension; j++) {
            std::cout << Query[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Key:" << std::endl;
    for (int i = 0; i < sequence_length; i++) {
        for (int j = 0; j < embed_dimension; j++) {
            std::cout << Key[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Value:" << std::endl;
    for (int i = 0; i < sequence_length; i++) {
        for (int j = 0; j < embed_dimension; j++) {
            std::cout << Value[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Output:" << std::endl;
    for (int i = 0; i < sequence_length; i++) {
        for (int j = 0; j < embed_dimension; j++) {
            std::cout << Output[i][j] << " ";
        }
        std::cout << std::endl;
    }

Error:
    cudaFree(device_Query);
    cudaFree(device_Key);
    cudaFree(device_Value);
    cudaFree(device_Output);
    cudaFree(device_max_values);
    cudaFree(device_sum_values);

    delete[] Query;
    delete[] Key;
    delete[] Value;
    delete[] Output;
    delete[] sum_values;
    delete[] max_values;

    return cudaStatus == cudaSuccess ? 0 : 1;
}
