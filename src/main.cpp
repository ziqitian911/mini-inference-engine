#include <iostream>
#include <vector>

class Tensor
{
private:
    float *data;
    std::vector<int> shape;
    int count;

public:
    Tensor(const std::vector<int> &shape)
    {
        this->shape = shape;
        this->count = 1;

        for (int num : shape)
        {
            this->count *= num;
        }

        this->data = new float[count];
    }

    ~Tensor()
    {
        delete[] this->data;
    }

    int size()
    {
        return count;
    }
};

int main(int argc, char const *argv[])
{

    Tensor t({2, 3});
    std::cout << "It compiles!" << std::endl;
    std::cout << "Hello World!" << std::endl;
    std::cout << t.size() << std::endl;

    return 0;
}
