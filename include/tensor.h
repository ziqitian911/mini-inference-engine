#pragma once
#include <vector>

class Tensor
{
private:
    int count;
    float *data;
    std::vector<int> shape;
    std::vector<int> strides;

public:
    Tensor(const std::vector<int> &shape);
    Tensor(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    Tensor(const std::vector<int> &shape, const std::vector<float> &values);
    ~Tensor();

    Tensor &operator=(const Tensor &other);
    Tensor &operator=(Tensor &&other) noexcept;

    int size() const;
    int dim(int i) const;

    float &at(const std::vector<int> &idx);
    float at(const std::vector<int> &idx) const;

    void print() const;
};