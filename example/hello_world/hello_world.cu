#include <stdio.h>

__global__ void printHelloFromGPU(int device, int totalDevices) {
    // Each thread will print the message from its assigned GPU
    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, totalDevices);
}

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA-capable GPU detected or CUDA error.\n");
        return 1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaSetDevice(device);

        // Launch the kernel on the current device
        printHelloFromGPU<<<1, 1>>>(device, deviceCount);
        
        // Synchronize to ensure the kernel has finished before moving to the next device
        cudaDeviceSynchronize();
    }

    return 0;
}

