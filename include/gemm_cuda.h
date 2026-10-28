#pragma once

void cuda_matmul(const float *a, const float *b, float *c, const int M, const int N, const int K);
void blocked_cuda_matmul(const float *a, const float *b, float *c, const int M, const int N, const int K);