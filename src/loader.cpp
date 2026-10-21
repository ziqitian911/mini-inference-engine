#include "loader.h"
#include <fstream>
#include <stdexcept>

Tensor load_tensor(const std::string &path, const std::vector<int> &shape)
{
    Tensor t(shape);

    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("cannot open " + path);
    }

    file.read(reinterpret_cast<char *>(t.begin()), t.size() * sizeof(float));

    return t;
}