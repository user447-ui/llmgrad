#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <string>

struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

struct Tensor : public std::enable_shared_from_this<Tensor> {
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<int> shape;
    std::vector<TensorPtr> _prev;
    std::function<void()> _backward;

    Tensor(std::vector<int> shape, std::vector<double> data = {});
    
    static TensorPtr zeros(int rows, int cols);
    static TensorPtr randn(int rows, int cols);
    static TensorPtr from_list(std::vector<int> shape, const std::vector<double>& list);

    // New Ops
    TensorPtr reshape(const std::vector<int>& new_shape);
    TensorPtr layernorm(TensorPtr gamma, TensorPtr beta, double eps = 1e-5);

    TensorPtr add(TensorPtr other);
    TensorPtr matmul(TensorPtr other);
    TensorPtr tanh();
    TensorPtr exp();
    TensorPtr log();
    TensorPtr sum();
    TensorPtr max();
    TensorPtr cross_entropy_loss(const std::vector<int>& targets);

    void backward();
    std::string to_string();
    int size() const;
    
    TensorPtr mul(TensorPtr other);
    TensorPtr relu();
};
