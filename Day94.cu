#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace torch;

// ---------------------
// Define a small CNN (ResNet-like)
// ---------------------
struct SmallResNetImpl : nn::Module {
    nn::Conv2d conv1{nullptr};
    nn::BatchNorm2d bn1{nullptr};
    nn::Sequential layer1, layer2, layer3;
    nn::Linear fc{nullptr};

    SmallResNetImpl(int64_t num_classes = 10) {
        conv1 = nn::Conv2d(nn::Conv2dOptions(3, 64, 3).stride(1).padding(1).bias(false));
        bn1 = nn::BatchNorm2d(64);

        // Simple residual blocks
        layer1 = nn::Sequential(
            nn::Conv2d(nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)),
            nn::BatchNorm2d(64),
            nn::ReLU(),
            nn::Conv2d(nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)),
            nn::BatchNorm2d(64)
        );

        layer2 = nn::Sequential(
            nn::Conv2d(nn::Conv2dOptions(64, 128, 3).stride(2).padding(1).bias(false)),
            nn::BatchNorm2d(128),
            nn::ReLU(),
            nn::Conv2d(nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)),
            nn::BatchNorm2d(128)
        );

        layer3 = nn::Sequential(
            nn::Conv2d(nn::Conv2dOptions(128, 256, 3).stride(2).padding(1).bias(false)),
            nn::BatchNorm2d(256),
            nn::ReLU(),
            nn::Conv2d(nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)),
            nn::BatchNorm2d(256)
        );

        fc = nn::Linear(256 * 8 * 8 / 16, num_classes); // adjust for downsampling
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1(conv1(x)));

        auto res = x;
        x = layer1->forward(x);
        x += res;
        x = torch::relu(x);

        x = layer2->forward(x);
        x = torch::relu(x);

        x = layer3->forward(x);
        x = torch::relu(x);

        x = torch::adaptive_avg_pool2d(x, {4, 4});
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        return x;
    }
};
TORCH_MODULE(SmallResNet);

// ---------------------
// Main Training Function
// ---------------------
int main() {
    torch::manual_seed(0);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << endl;

    const int64_t batch_size = 128;
    const int64_t epochs = 10;
    const double lr = 0.01;

    // CIFAR10 data loader (automatically downloads if missing)
    auto train_dataset = torch::data::datasets::CIFAR10("./data")
                             .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                                       {0.2023, 0.1994, 0.2010}))
                             .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_dataset = torch::data::datasets::CIFAR10("./data", torch::data::datasets::CIFAR10::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                                      {0.2023, 0.1994, 0.2010}))
                            .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), 256);

    SmallResNet model(10);
    model->to(device);

    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(lr).momentum(0.9).weight_decay(5e-4));

    torch::cuda::amp::GradScaler scaler;  // Mixed precision scaler
    auto loss_fn = torch::nn::CrossEntropyLoss();

    cout << "Training..." << endl;
    auto start = chrono::steady_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        int64_t num_samples = 0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            optimizer.zero_grad();

            // Mixed precision context
            torch::cuda::amp::autocast_mode autocast_guard(true);
            auto output = model->forward(data);
            auto loss = loss_fn(output, target);

            scaler.scale(loss).backward();
            scaler.step(optimizer);
            scaler.update();

            epoch_loss += loss.item<double>() * data.size(0);
            num_samples += data.size(0);
        }

        double avg_loss = epoch_loss / num_samples;
        cout << "Epoch [" << epoch << "/" << epochs << "] Loss: " << avg_loss << endl;
    }

    auto end = chrono::steady_clock::now();
    auto dur = chrono::duration_cast<chrono::seconds>(end - start).count();
    cout << "Training finished in " << dur << " seconds." << endl;

    // Save model
    torch::save(model, "day94_resnet_cifar10.pt");
    cout << "Model saved to day94_resnet_cifar10.pt" << endl;
}
