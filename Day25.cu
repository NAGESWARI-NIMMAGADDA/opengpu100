#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define TOTAL_ROLLOUTS 1024   // Number of simulations per node
#define MAX_ROLLOUT_DEPTH 100 // Maximum steps per rollout

// Define the game state 
struct GameState {
    int available_moves[10];  
    int move_count;
    bool is_terminal_state;
    float accumulated_reward;  // Reward if terminal

    __device__ GameState compute_next_state(int chosen_action) {
        GameState new_state = *this;
        // Apply the action 
        new_state.accumulated_reward += (chosen_action % 2 == 0) ? 1.0f : -1.0f;
        new_state.is_terminal_state = (new_state.accumulated_reward > 10 || new_state.accumulated_reward < -10);
        return new_state;
    }

    __device__ int select_random_action(curandState* rng_state) {
        if (move_count == 0) return -1;
        return available_moves[curand(rng_state) % move_count];
    }
};

// Node structure for MCTS
struct MCTSNode {
    GameState game_state;
    int visit_count;
    float node_value;
};

// Device function for rollout (Simulation phase)
__device__ float perform_rollout(GameState state, curandState* rng_state) {
    int depth_counter = 0;
    while (!state.is_terminal_state && depth_counter < MAX_ROLLOUT_DEPTH) {
        int chosen_action = state.select_random_action(rng_state);
        if (chosen_action == -1) break;  // No moves available

        state = state.compute_next_state(chosen_action);
        depth_counter++;

        // Debug: Print rollout step info
        printf("[Thread %d] Rollout step %d | Action: %d | Reward: %.2f\n", 
               threadIdx.x + blockIdx.x * blockDim.x,
               depth_counter, chosen_action, state.accumulated_reward);
    }
    return state.accumulated_reward;
}

// Kernel to run parallel rollouts
__global__ void mcts_simulation_kernel(MCTSNode* device_nodes, int total_nodes, float* result_values) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index >= total_nodes) return;

    curandState rng_state;
    curand_init(thread_index, 0, 0, &rng_state);

    float sum_rewards = 0.0f;
    for (int sim = 0; sim < TOTAL_ROLLOUTS; sim++) {
        sum_rewards += perform_rollout(device_nodes[thread_index].game_state, &rng_state);
    }

    float avg_reward = sum_rewards / TOTAL_ROLLOUTS;
    result_values[thread_index] = avg_reward;

    // Debug: Print final node result
    printf("[Thread %d] Final average reward: %.4f\n", thread_index, avg_reward);
}

// Host function to execute MCTS
void execute_mcts(MCTSNode* host_nodes, int total_nodes) {
    MCTSNode* device_nodes;
    float* device_results;
    float* host_results = (float*)malloc(total_nodes * sizeof(float));

    cudaMalloc(&device_nodes, total_nodes * sizeof(MCTSNode));
    cudaMalloc(&device_results, total_nodes * sizeof(float));

    cudaMemcpy(device_nodes, host_nodes, total_nodes * sizeof(MCTSNode), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (total_nodes + threadsPerBlock - 1) / threadsPerBlock;

    printf("[HOST] Launching kernel with %d blocks and %d threads...\n", blocksPerGrid, threadsPerBlock);
    mcts_simulation_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_nodes, total_nodes, device_results);

    cudaDeviceSynchronize(); // Ensure all prints finish

    cudaMemcpy(host_results, device_results, total_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    // Update values in host nodes
    for (int i = 0; i < total_nodes; i++) {
        host_nodes[i].node_value = host_results[i];
        printf("[HOST] Node %d final value: %.4f\n", i, host_nodes[i].node_value);
    }

    free(host_results);
    cudaFree(device_nodes);
    cudaFree(device_results);
}

int main() {
    // Create root node with example state
    MCTSNode root_node;
    root_node.game_state.move_count = 10;
    root_node.game_state.is_terminal_state = false;
    root_node.game_state.accumulated_reward = 0.0f;
    for (int i = 0; i < 10; i++) {
        root_node.game_state.available_moves[i] = i;
    }
    root_node.visit_count = 0;
    root_node.node_value = 0.0f;

    printf("[HOST] Starting MCTS...\n");
    execute_mcts(&root_node, 1);

    printf("[HOST] MCTS result for root node: %.4f\n", root_node.node_value);
    return 0;
}
