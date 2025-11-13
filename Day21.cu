#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

// CUDA kernel to compute predictions and squared loss
__global__ void compute_loss(float* features, float* labels, float* weights, float* bias,
                             float* loss_array, float* predictions, int num_samples, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {  
        float prediction_value = 0.0f;
        for (int f = 0; f < num_features; f++) {
            prediction_value += features[idx * num_features + f] * weights[f];
        }
        prediction_value += *bias; // scalar bias
        predictions[idx] = prediction_value;
        float error = labels[idx] - prediction_value;
        loss_array[idx] = error * error; // squared loss
    }
}

// CUDA kernel to compute gradients
__global__ void compute_gradients(float* features, float* loss_array, float* grad_weights, float* grad_bias,
                                  int num_samples, int num_features) {
    __shared__ float shared_grad_bias[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_features) {
        float gradW = 0.0f;
        for (int i = 0; i < num_samples; i++) {
            gradW += features[i * num_features + idx] * loss_array[i];
        }
        grad_weights[idx] = - (2.0f / num_samples) * gradW;
    }

    float gradb = 0.0f;
    if (idx < num_samples) {
        gradb = loss_array[idx];
    }
    shared_grad_bias[threadIdx.x] = gradb;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum_gradb = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum_gradb += shared_grad_bias[i];
        }
        atomicAdd(grad_bias, - (2.0f / num_samples) * sum_gradb);
    }
}

// CUDA kernel to update weights
__global__ void update_weights(float* weights, float* grad_weights, float* bias, float* grad_bias,
                               float learning_rate, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_features) {
        weights[idx] -= learning_rate * grad_weights[idx];
    }
    if (idx == 0) {
        *bias -= learning_rate * (*grad_bias);
    }
}

// Host function to train the model
void train_sgd(float* host_features, float* host_labels, float* host_weights, float* host_bias,
               int num_samples, int num_features, float learning_rate, int epochs) {
    float *dev_features, *dev_labels, *dev_weights, *dev_bias;
    float *dev_grad_weights, *dev_grad_bias, *dev_loss_array, *dev_predictions;

    cudaMalloc(&dev_features, num_samples * num_features * sizeof(float));
    cudaMalloc(&dev_labels, num_samples * sizeof(float));
    cudaMalloc(&dev_weights, num_features * sizeof(float));
    cudaMalloc(&dev_bias, sizeof(float));
    cudaMalloc(&dev_grad_weights, num_features * sizeof(float));
    cudaMalloc(&dev_grad_bias, sizeof(float));
    cudaMalloc(&dev_loss_array, num_samples * sizeof(float));
    cudaMalloc(&dev_predictions, num_samples * sizeof(float));
    
    cudaMemcpy(dev_features, host_features, num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_labels, host_labels, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weights, host_weights, num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bias, host_bias, sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks_samples = (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_features = (num_features + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        compute_loss<<<blocks_samples, BLOCK_SIZE>>>(dev_features, dev_labels, dev_weights, dev_bias,
                                                     dev_loss_array, dev_predictions, num_samples, num_features);
        cudaDeviceSynchronize();

        compute_gradients<<<blocks_features, BLOCK_SIZE>>>(dev_features, dev_loss_array,
                                                           dev_grad_weights, dev_grad_bias,
                                                           num_samples, num_features);
        cudaDeviceSynchronize();

        update_weights<<<blocks_features, BLOCK_SIZE>>>(dev_weights, dev_grad_weights,
                                                        dev_bias, dev_grad_bias,
                                                        learning_rate, num_features);
        cudaDeviceSynchronize();

        // Debug: print loss every 100 epochs
        if (epoch % 100 == 0 || epoch == epochs - 1) {
            float* host_loss_array = new float[num_samples];
            cudaMemcpy(host_loss_array, dev_loss_array, num_samples * sizeof(float), cudaMemcpyDeviceToHost);
            float total_loss = 0.0f;
            for (int i = 0; i < num_samples; i++) total_loss += host_loss_array[i];
            std::cout << "Epoch " << epoch << " - Loss: " << total_loss / num_samples << std::endl;
            delete[] host_loss_array;
        }
    }

    cudaMemcpy(host_weights, dev_weights, num_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_bias, dev_bias, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_features);
    cudaFree(dev_labels);
    cudaFree(dev_weights);
    cudaFree(dev_bias);
    cudaFree(dev_grad_weights);
    cudaFree(dev_grad_bias);
    cudaFree(dev_loss_array);
    cudaFree(dev_predictions);
}

int main() {
    int num_samples = 1024;
    int num_features = 10;
    float learning_rate = 0.01;
    int epochs = 1000;

    float *host_features = new float[num_samples * num_features];
    float *host_labels = new float[num_samples];
    float *host_weights = new float[num_features];
    float *host_bias = new float[1];

    srand(42);
    for (int i = 0; i < num_samples * num_features; i++) {
        host_features[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < num_samples; i++) {
        host_labels[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < num_features; i++) {
        host_weights[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    *host_bias = static_cast<float>(rand()) / RAND_MAX;

    std::cout << "Starting training...\n";
    train_sgd(host_features, host_labels, host_weights, host_bias,
              num_samples, num_features, learning_rate, epochs);

    std::cout << "\nFinal trained weights: ";
    for (int i = 0; i < num_features; i++) std::cout << host_weights[i] << " ";
    std::cout << "\nFinal trained bias: " << *host_bias << std::endl;

    delete[] host_features;
    delete[] host_labels;
    delete[] host_weights;
    delete[] host_bias;
    return 0;
}
