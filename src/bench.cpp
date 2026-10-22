#include <iostream>
#include <chrono>
#include <vector>

#include "tensor.h"
#include "gemm.h"

int main(int argc, char const *argv[])
{
    const int s = 2048;
    std::vector<float> values(s * s, 1);

    Tensor A({s, s}, values), B({s, s}, values);

    auto t0 = std::chrono::steady_clock::now();
    Tensor C = matmul(A, B, false);
    auto t1 = std::chrono::steady_clock::now();

    auto t3 = std::chrono::steady_clock::now();
    Tensor D = matmul(A, B);
    auto t4 = std::chrono::steady_clock::now();

    double sec_norm = std::chrono::duration<double>(t1 - t0).count();
    double sec_reorder = std::chrono::duration<double>(t4 - t3).count();

    std::cout << "Normal: " << sec_norm << "\n";
    std::cout << "Reorder: " << sec_reorder << "\n";

    return 0;
}
