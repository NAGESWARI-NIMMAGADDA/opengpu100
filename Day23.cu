#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>

// Kernel function for SwiGLU
__global__ void swiglu_kernel(float* output_data, const float* input_data, const float* weights_gate, const float* weights_value, int batch_size, int hidden_dim, int output_dim) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && o < output_dim) {
        float gate_sum = 0.0f;
        float value_sum = 0.0f;
        
        for (int i = 0; i < hidden_dim; i++) {
            gate_sum  += input_data[b * hidden_dim + i] * weights_gate[o + i * output_dim];
            value_sum += input_data[b * hidden_dim + i] * weights_value[o + i * output_dim];
        }
        
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_sum));
        float result = gate_sum * sigmoid_val * value_sum;
        
        // Print debug info for the first thread
        if (b == 0 && o == 0) {
            printf("GPU Debug (b=0, o=0): gate_sum=%f, value_sum=%f, sigmoid_val=%f, result=%f\n",
                   gate_sum, value_sum, sigmoid_val, result);
        }
        
        output_data[b * output_dim + o] = result;
    }
}

void swiglu_forward(float* output_data, const float* input_data, const float* weights_gate, const float* weights_value, int batch_size, int hidden_dim, int output_dim) {
    // Allocate memory on GPU
    float *d_input, *d_gate, *d_value, *d_output;
    cudaMalloc((void**)&d_input, batch_size * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_gate, hidden_dim * output_dim * sizeof(float));
    cudaMalloc((void**)&d_value, hidden_dim * output_dim * sizeof(float));
    cudaMalloc((void**)&d_output, batch_size * output_dim * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_input, input_data, batch_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, weights_gate, hidden_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, weights_value, hidden_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define CUDA kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch kernel
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input, d_gate, d_value, batch_size, hidden_dim, output_dim);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy result back to CPU
    cudaMemcpy(output_data, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_gate);
    cudaFree(d_value);
    cudaFree(d_output);
}

int main() {
    int batch_size = 32;
    int hidden_dim = 128;
    int output_dim = 64;
    
    // Allocate memory
    float *input_data = new float[batch_size * hidden_dim];
    float *weights_gate = new float[hidden_dim * output_dim];
    float *weights_value = new float[hidden_dim * output_dim];
    float *swiglu_output = new float[batch_size * output_dim];
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Initialize input and weights
    for (int i = 0; i < batch_size * hidden_dim; i++) {
        input_data[i] = dis(gen);
    }
    for (int i = 0; i < hidden_dim * output_dim; i++) {
        weights_gate[i]  = dis(gen);
        weights_value[i] = dis(gen);
    }
    
    // Manual CPU calculation for first element (for verification)
    float manual_gate_sum = 0.0f;
    float manual_value_sum = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        manual_gate_sum  += input_data[i] * weights_gate[i * output_dim];
        manual_value_sum += input_data[i] * weights_value[i * output_dim];
    }
    float manual_sigmoid = 1.0f / (1.0f + exp(-manual_gate_sum));
    float manual_result = manual_gate_sum * manual_sigmoid * manual_value_sum;
    
    std::cout << "===== CPU Manual Calculation (First Output Element) =====" << std::endl;
    std::cout << "Gate Sum: " << manual_gate_sum << std::endl;
    std::cout << "Value Sum: " << manual_value_sum << std::endl;
    std::cout << "Sigmoid:   " << manual_sigmoid << std::endl;
    std::cout << "Expected:  " << manual_result << std::endl;
    
    // Compute SwiGLU on GPU
    swiglu_forward(swiglu_output, input_data, weights_gate, weights_value, batch_size, hidden_dim, output_dim);
    
    // Print sample data
    std::cout << "\n===== First 10 Input Values =====" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "input_data[" << i << "] = " << input_data[i] << std::endl;
    }
    
    std::cout << "\n===== First 10 Gate Weights =====" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "weights_gate[" << i << "] = " << weights_gate[i] << std::endl;
    }
    
    std::cout << "\n===== First 10 Value Weights =====" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "weights_value[" << i << "] = " << weights_value[i] << std::endl;
    }
    
    std::cout << "\n===== First 10 Output Values =====" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "swiglu_output[" << i << "] = " << swiglu_output[i] << std::endl;
    }
    
    // Free memory
    delete[] input_data;
    delete[] weights_gate;
    delete[] weights_value;
    delete[] swiglu_output;
    
    return 0;
}
