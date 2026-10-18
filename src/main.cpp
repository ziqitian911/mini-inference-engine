#include <iostream>
#include "tensor.h"
#include "gemm.h"
#include "ops.h"

int main(int argc, char const *argv[])
{
    // Tensor t({3, 2}, {1, 2, 3, 4, 5, 6});
    // t.at({1, 2}) = 99;
    // Tensor t2 = t;
    // std::cout << t.at({1, 2}) << std::endl;
    // std::cout << t2.at({1, 2}) << std::endl;

    // Tensor A({5, 1}, {1, 2, 3, 2, 1});
    // Tensor B({1, 5}, {1, 2, 3, 2, 1});

    // Tensor C = matmul(A, B);
    // C.print();

    // std::cout << C.at({1, 2}) << std::endl;

    // std::cout << "\n";

    // std::cout << "ReLU" << std::endl;

    // Tensor x({2, 3}, {-1, 2, -3, 4, -5, 6});
    // Tensor y = relu(x);
    // y.print();

    // std::cout << "\n";

    // std::cout << "Linear" << "\n";

    // Tensor x({1, 3}, {1, 2, 3});
    // Tensor W({3, 2}, {1, 2, 3, 4, 5, 6});
    // Tensor b({1, 2}, {10, 20});

    // Tensor out = linear(x, W, b);
    // out.print();

    // std::cout << "Softmax\n";

    // Tensor s = softmax(out);

    // s.print();

    Tensor s({1, 3}, {1, 2, 3});
    Tensor p = softmax(s);
    p.print();

    return 0;
}
