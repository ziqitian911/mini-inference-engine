#include "gemm_cuda.h"

__global__ void mtx_mul(float *a, float *b, float *c, const int M, const int N, const int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        c[row * N + col] = 0.0f;
        for (int i = 0; i < K; i++)
        {
            c[row * N + col] += a[row * K + i] * b[i * N + col];
        }
    }
}

void cuda_matmul(const float *a, const float *b, float *c, const int M, const int N, const int K)
{

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    mtx_mul<<<blocks, threads>>>(d_a, d_b, d_c, M, N, K);

    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}