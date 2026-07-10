#include <stdio.h>
#include <stdlib.h>

void setGPU(int gpu_id) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No GPU device available\n");
        exit(1);
    }
    cudaSetDevice(gpu_id);
}