#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    std::cout << "Initializing cuDNN..." << std::endl;

    // Define dimensions (example sizes)
    int input_n = 64, input_c = 1, input_h = 28, input_w = 28;
    int hidden1_c = 16, hidden2_c = 32, output_c = 10;

    // Create descriptors
    cudnnTensorDescriptor_t input_descriptor, hidden1_descriptor, hidden2_descriptor, output_descriptor;
    cudnnFilterDescriptor_t filter1_descriptor, filter2_descriptor, filter3_descriptor;
    cudnnConvolutionDescriptor_t conv1_descriptor, conv2_descriptor, conv3_descriptor;

    // Create tensor descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&hidden1_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&hidden2_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));

    // Create filter descriptors
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter1_descriptor));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter2_descriptor));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter3_descriptor));

    // Create convolution descriptors
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv1_descriptor));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv2_descriptor));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv3_descriptor));

    std::cout << "Setting tensor descriptors..." << std::endl;

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           input_n, input_c, input_h, input_w));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter1_descriptor,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           hidden1_c, input_c, 3, 3));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv1_descriptor,
                                                1, 1, 1, 1, 1, 1,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Set output tensor dimensions manually (example only)
    int hidden1_h = input_h, hidden1_w = input_w;
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(hidden1_descriptor,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           input_n, hidden1_c, hidden1_h, hidden1_w));

    std::cout << "Setting filter descriptors..." << std::endl;

    // Allocate memory
    std::cout << "Allocating memory for data and parameters..." << std::endl;

    float *d_input, *d_hidden1, *d_output, *d_filter1;
    size_t input_bytes = input_n * input_c * input_h * input_w * sizeof(float);
    size_t hidden1_bytes = input_n * hidden1_c * hidden1_h * hidden1_w * sizeof(float);
    size_t filter1_bytes = hidden1_c * input_c * 3 * 3 * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_hidden1, hidden1_bytes));
    CHECK_CUDA(cudaMalloc(&d_filter1, filter1_bytes));

    // Initialize input and filter (optional)
    std::cout << "Initializing weights and inputs..." << std::endl;
    CHECK_CUDA(cudaMemset(d_input, 1, input_bytes));
    CHECK_CUDA(cudaMemset(d_filter1, 1, filter1_bytes));

    // Set scalars and workspace
    float alpha_val = 1.0f;
    float beta_val = 0.0f;
    const void* alpha = static_cast<const void*>(&alpha_val);
    const void* beta = static_cast<const void*>(&beta_val);

    void* workspace = nullptr;
    size_t workspace_bytes = 0;

    // Get workspace size
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        filter1_descriptor,
        conv1_descriptor,
        hidden1_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        &workspace_bytes));

    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));

    std::cout << "Starting training loop..." << std::endl;

    for (int epoch = 0; epoch < 10; ++epoch) {
        std::cout << "\nEpoch " << epoch + 1 << "/10" << std::endl;
        std::cout << "  Forward pass: Input → Hidden1" << std::endl;

        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            alpha,
            input_descriptor,
            d_input,
            filter1_descriptor,
            d_filter1,
            conv1_descriptor,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            workspace,
            workspace_bytes,
            beta,
            hidden1_descriptor,
            d_hidden1));

        // Add other layers here similarly: Hidden1 → Hidden2 → Output

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaPeekAtLastError());
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudaFree(d_input);
    cudaFree(d_hidden1);
    cudaFree(d_filter1);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(hidden1_descriptor);
    cudnnDestroyFilterDescriptor(filter1_descriptor);
    cudnnDestroyConvolutionDescriptor(conv1_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}

