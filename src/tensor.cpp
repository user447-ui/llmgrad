#include "tensor.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <iostream>

int Tensor::size() const {
    int s = 1;
    for(int d : shape) s *= d;
    return s;
}

Tensor::Tensor(std::vector<int> s, std::vector<double> d) : shape(s) {
    if (d.empty()) data.resize(size(), 0.0);
    else data = d;
    grad.resize(size(), 0.0);
    _backward = [](){};
}

TensorPtr Tensor::zeros(int rows, int cols) {
    return std::make_shared<Tensor>(std::vector<int>{rows, cols});
}

TensorPtr Tensor::randn(int rows, int cols) {
    auto t = zeros(rows, cols);
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 1.0/std::sqrt(cols)); 
    for(auto& v : t->data) v = dist(gen);
    return t;
}

TensorPtr Tensor::from_list(int rows, int cols, const std::vector<double>& list) {
    return std::make_shared<Tensor>(std::vector<int>{rows, cols}, list);
}

TensorPtr Tensor::add(TensorPtr other) {
    auto out = zeros(shape[0], shape[1]);
    bool broadcast = (other->shape[0] == 1);
    for (int i = 0; i < size(); ++i) {
        int col = i % shape[1];
        out->data[i] = data[i] + (broadcast ? other->data[col] : other->data[i]);
    }
    out->_prev = {shared_from_this(), other};
    out->_backward = [out, this_ptr=shared_from_this(), other, broadcast]() {
        for (int i = 0; i < out->size(); ++i) {
            this_ptr->grad[i] += out->grad[i];
            if(broadcast) other->grad[i % out->shape[1]] += out->grad[i];
            else other->grad[i] += out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::matmul(TensorPtr other) {
    int M = shape[0], K = shape[1], N = other->shape[1];
    auto out = zeros(M, N);
    for (int m = 0; m < M; ++m) 
        for (int n = 0; n < N; ++n) 
            for (int k = 0; k < K; ++k) 
                out->data[m*N+n] += data[m*K+k] * other->data[k*N+n];
    
    out->_prev = {shared_from_this(), other};
    out->_backward = [out, this_ptr=shared_from_this(), other, M, K, N]() {
        for (int m=0; m<M; ++m)
            for (int n=0; n<N; ++n) {
                 double g = out->grad[m*N+n];
                 for (int k=0; k<K; ++k) {
                     this_ptr->grad[m*K+k] += g * other->data[k*N+n];
                     other->grad[k*N+n] += this_ptr->data[m*K+k] * g;
                 }
            }
    };
    return out;
}

TensorPtr Tensor::tanh() {
    auto out = zeros(shape[0], shape[1]);
    for(int i=0; i<size(); ++i) out->data[i] = std::tanh(data[i]);
    out->_prev = {shared_from_this()};
    
    out->_backward = [out, this_ptr=shared_from_this()]() {
        for(int i=0; i<out->size(); ++i) {
            this_ptr->grad[i] += (1.0 - out->data[i]*out->data[i]) * out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::cross_entropy_loss(const std::vector<int>& targets) {
    int batch = shape[0], classes = shape[1];
    auto out = zeros(1, 1);
    auto probs = std::make_shared<std::vector<double>>(data.size());
    double total_loss = 0;
    
    for(int b=0; b<batch; ++b) {
        double mx = -1e9;
        for(int c=0; c<classes; ++c) if(data[b*classes+c] > mx) mx = data[b*classes+c];
        double sum = 0;
        for(int c=0; c<classes; ++c) {
            double e = std::exp(data[b*classes+c] - mx);
            (*probs)[b*classes+c] = e;
            sum += e;
        }
        for(int c=0; c<classes; ++c) (*probs)[b*classes+c] /= sum;
        total_loss -= std::log((*probs)[b*classes + targets[b]] + 1e-7);
    }
    out->data[0] = total_loss / batch;
    out->_prev = {shared_from_this()};
    out->_backward = [out, this_ptr=shared_from_this(), targets, probs, batch, classes]() {
        double g = out->grad[0] / batch;
        for(int b=0; b<batch; ++b) {
            int t = targets[b];
            for(int c=0; c<classes; ++c) {
                this_ptr->grad[b*classes+c] += ((*probs)[b*classes+c] - (c==t?1.0:0.0)) * g;
            }
        }
    };
    return out;
}

// Stubs
TensorPtr Tensor::mul(TensorPtr o) { return zeros(1,1); } 
TensorPtr Tensor::relu() { return zeros(1,1); }
TensorPtr Tensor::log() { return zeros(1,1); }
TensorPtr Tensor::exp() { return zeros(1,1); }
TensorPtr Tensor::sum() { return zeros(1,1); }
TensorPtr Tensor::max() { return zeros(1,1); }

void Tensor::backward() {
    std::vector<TensorPtr> topo;
    std::set<TensorPtr> visited;
    std::function<void(TensorPtr)> build = [&](TensorPtr v) {
        if(!visited.count(v)) { visited.insert(v); for(auto p : v->_prev) build(p); topo.push_back(v); }
    };
    build(shared_from_this());
    std::fill(grad.begin(), grad.end(), 1.0);
    std::reverse(topo.begin(), topo.end());
    for(auto v : topo) v->_backward();
}
std::string Tensor::to_string() { return "Tensor(" + std::to_string(shape[0]) + "x" + std::to_string(shape[1]) + ")"; }
