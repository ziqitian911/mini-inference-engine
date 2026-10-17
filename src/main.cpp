#include <iostream>
#include "tensor.h"
#include "gemm.h"

int main(int argc, char const *argv[])
{
    // Tensor t({3, 2}, {1, 2, 3, 4, 5, 6});
    // t.at({1, 2}) = 99;
    // Tensor t2 = t;
    // std::cout << t.at({1, 2}) << std::endl;
    // std::cout << t2.at({1, 2}) << std::endl;

    Tensor A({5, 1}, {1, 2, 3, 2, 1});
    Tensor B({1, 5}, {1, 2, 3, 2, 1});

    Tensor C = matmul(A, B);
    C.print();

    std::cout << C.at({1, 2}) << std::endl;

    return 0;
}
