#include <iostream>

__global__ void vec_add(float *a, float *b, float *c, int n)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char const *argv[])
{

    int n = 10;

    float h_a[10], h_b[10];

    for (int i = 0; i < n; i++)
    {
        h_a[i] = 1, h_b[i] = i;
    }

    float h_c[10];

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
    {
        std::cout << h_c[i] << " ";
    }

    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
