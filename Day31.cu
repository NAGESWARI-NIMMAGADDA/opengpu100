#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_X 16
#define TILE_Y 16

__global__ void gameOfLifeKernel(const int *gridIn, int *gridOut, int WIDTH, int HEIGHT) {
    extern __shared__ int sharedTile[];

    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * bdx + tx;
    int gy = blockIdx.y * bdy + ty;

    int sWidth = bdx + 2;
    int sx = tx + 1;
    int sy = ty + 1;

    // Load the current cell into shared memory
    if (gx < WIDTH && gy < HEIGHT)
        sharedTile[sy * sWidth + sx] = gridIn[gy * WIDTH + gx];
    else
        sharedTile[sy * sWidth + sx] = 0;

    // Load halos
    if (ty == 0) {
        int gY = gy - 1;
        int sY_top = 0;
        if (gY >= 0 && gx < WIDTH)
            sharedTile[sY_top * sWidth + sx] = gridIn[gY * WIDTH + gx];
        else
            sharedTile[sY_top * sWidth + sx] = 0;
    }
    if (ty == bdy - 1) {
        int gY = gy + 1;
        int sY_bottom = bdy + 1;
        if (gY < HEIGHT && gx < WIDTH)
            sharedTile[sY_bottom * sWidth + sx] = gridIn[gY * WIDTH + gx];
        else
            sharedTile[sY_bottom * sWidth + sx] = 0;
    }
    if (tx == 0) {
        int gX = gx - 1;
        int sX_left = 0;
        if (gX >= 0 && gy < HEIGHT)
            sharedTile[sy * sWidth + sX_left] = gridIn[gy * WIDTH + gX];
        else
            sharedTile[sy * sWidth + sX_left] = 0;
    }
    if (tx == bdx - 1) {
        int gX = gx + 1;
        int sX_right = bdx + 1;
        if (gX < WIDTH && gy < HEIGHT)
            sharedTile[sy * sWidth + sX_right] = gridIn[gy * WIDTH + gX];
        else
            sharedTile[sy * sWidth + sX_right] = 0;
    }
    if (tx == 0 && ty == 0) {
        int gX = gx - 1, gY = gy - 1;
        int sIdx = 0;
        if (gX >= 0 && gY >= 0)
            sharedTile[sIdx] = gridIn[gY * WIDTH + gX];
        else
            sharedTile[sIdx] = 0;
    }
    if (tx == bdx - 1 && ty == 0) {
        int gX = gx + 1, gY = gy - 1;
        int sIdx = (0 * sWidth) + (bdx + 1);
        if (gX < WIDTH && gY >= 0)
            sharedTile[sIdx] = gridIn[gY * WIDTH + gX];
        else
            sharedTile[sIdx] = 0;
    }
    if (tx == 0 && ty == bdy - 1) {
        int gX = gx - 1, gY = gy + 1;
        int sIdx = (bdy + 1) * sWidth + 0;
        if (gX >= 0 && gY < HEIGHT)
            sharedTile[sIdx] = gridIn[gY * WIDTH + gX];
        else
            sharedTile[sIdx] = 0;
    }
    if (tx == bdx - 1 && ty == bdy - 1) {
        int gX = gx + 1, gY = gy + 1;
        int sIdx = (bdy + 1) * sWidth + (bdx + 1);
        if (gX < WIDTH && gY < HEIGHT)
            sharedTile[sIdx] = gridIn[gY * WIDTH + gX];
        else
            sharedTile[sIdx] = 0;
    }

    __syncthreads();

    if (gx < WIDTH && gy < HEIGHT) {
        int neighborSum = 0;
        neighborSum += sharedTile[(sy - 1) * sWidth + (sx - 1)];
        neighborSum += sharedTile[(sy - 1) * sWidth + (sx)];
        neighborSum += sharedTile[(sy - 1) * sWidth + (sx + 1)];
        neighborSum += sharedTile[(sy) * sWidth + (sx - 1)];
        neighborSum += sharedTile[(sy) * sWidth + (sx + 1)];
        neighborSum += sharedTile[(sy + 1) * sWidth + (sx - 1)];
        neighborSum += sharedTile[(sy + 1) * sWidth + (sx)];
        neighborSum += sharedTile[(sy + 1) * sWidth + (sx + 1)];

        int cellState = sharedTile[sy * sWidth + sx];
        int newState = 0;

        if (cellState == 1 && (neighborSum == 2 || neighborSum == 3))
            newState = 1;
        else if (cellState == 0 && neighborSum == 3)
            newState = 1;

        gridOut[gy * WIDTH + gx] = newState;

        // Debug print (only from a few threads to reduce spam)
        if (gx < 4 && gy < 4) {
            printf("Thread (%d,%d): old=%d, neighbors=%d, new=%d\n",
                   gx, gy, cellState, neighborSum, newState);
        }
    }
}

int main() {
    const int WIDTH = 16;
    const int HEIGHT = 16;
    const int GRID_SIZE = WIDTH * HEIGHT;

    int *hostGridIn  = (int*)malloc(GRID_SIZE * sizeof(int));
    int *hostGridOut = (int*)malloc(GRID_SIZE * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < GRID_SIZE; i++) {
        hostGridIn[i] = rand() % 2;
    }

    printf("Initial Grid:\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%d ", hostGridIn[y * WIDTH + x]);
        }
        printf("\n");
    }
    printf("\n");

    int *devGridIn, *devGridOut;
    cudaMalloc(&devGridIn, GRID_SIZE * sizeof(int));
    cudaMalloc(&devGridOut, GRID_SIZE * sizeof(int));

    cudaMemcpy(devGridIn, hostGridIn, GRID_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(TILE_X, TILE_Y);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    size_t sharedMemSize = (block.x + 2) * (block.y + 2) * sizeof(int);

    printf("Launching kernel with grid=(%d,%d), block=(%d,%d), shared=%zu bytes\n",
           grid.x, grid.y, block.x, block.y, sharedMemSize);

    gameOfLifeKernel<<<grid, block, sharedMemSize>>>(devGridIn, devGridOut, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(hostGridOut, devGridOut, GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nGrid After One Iteration:\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%d ", hostGridOut[y * WIDTH + x]);
        }
        printf("\n");
    }

    cudaFree(devGridIn);
    cudaFree(devGridOut);
    free(hostGridIn);
    free(hostGridOut);

    return 0;
}
