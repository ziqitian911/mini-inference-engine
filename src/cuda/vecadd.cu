#include <iostream>

__global__ void myKernel()
{
    printf("Hello from CUDA\n");
}

int main(int argc, char const *argv[])
{
    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
