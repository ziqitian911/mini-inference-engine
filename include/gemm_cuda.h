#pragma once

float cuda_matmul(const float *a, const float *b, float *c, const int M, const int N, const int K);
float blocked_cuda_matmul(const float *a, const float *b, float *c, const int M, const int N, const int K);