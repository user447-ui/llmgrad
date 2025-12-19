#include "tensor.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <iostream>
#include <numeric>

// --- Utils ---
int Tensor::size() const {
    int s = 1;
    for(int d : shape) s *= d;
    return s;
}

// --- Constructor ---
Tensor::Tensor(std::vector<int> s, std::vector<double> d) : shape(s) {
    if (d.empty()) data.resize(size(), 0.0);
    else {
        if (d.size() != size()) throw std::runtime_error("Data size mismatch in constructor");
        data = d;
    }
    grad.resize(size(), 0.0);
    _backward = [](){};
}

// --- Factories ---
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

TensorPtr Tensor::from_list(std::vector<int> shape, const std::vector<double>& list) {
    return std::make_shared<Tensor>(shape, list);
}

// --- Core Ops ---

TensorPtr Tensor::reshape(const std::vector<int>& new_shape) {
    int new_size = 1;
    for(int s : new_shape) new_size *= s;
    if (new_size != size()) throw std::runtime_error("Reshape: total elements mismatch");

    auto out = std::make_shared<Tensor>(new_shape, data);
    
    out->_prev = {shared_from_this()};
    out->_backward = [out, this_ptr=shared_from_this()]() {
        for(int i=0; i<out->size(); ++i) {
            this_ptr->grad[i] += out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::add(TensorPtr other) {
    auto out = std::make_shared<Tensor>(shape);
    bool broadcast = (other->size() < size());
    
    int cols = shape.back();
    
    for (int i = 0; i < size(); ++i) {
        double b_val = broadcast ? other->data[i % cols] : other->data[i];
        out->data[i] = data[i] + b_val;
    }
    
    out->_prev = {shared_from_this(), other};
    out->_backward = [out, this_ptr=shared_from_this(), other, broadcast, cols]() {
        for (int i = 0; i < out->size(); ++i) {
            this_ptr->grad[i] += out->grad[i];
            if(broadcast) other->grad[i % cols] += out->grad[i];
            else other->grad[i] += out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::matmul(TensorPtr other) {
    if(shape.size() != 2 || other->shape.size() != 2) 
        throw std::runtime_error("Matmul currently supports only 2D tensors. Use reshape.");

    int M = shape[0], K = shape[1], N = other->shape[1];
    auto out = zeros(M, N);
    
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; ++m) 
        for (int n = 0; n < N; ++n) {
            double sum = 0;
            for (int k = 0; k < K; ++k) sum += data[m*K+k] * other->data[k*N+n];
            out->data[m*N+n] = sum;
        }
    
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

// Fused LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
TensorPtr Tensor::layernorm(TensorPtr gamma, TensorPtr beta, double eps) {
    int batch = shape[0];
    int emb = shape[1];
    
    auto out = std::make_shared<Tensor>(shape);
    
    auto cache_mean = std::make_shared<std::vector<double>>(batch);
    auto cache_rstd = std::make_shared<std::vector<double>>(batch);
    
    for(int b=0; b<batch; ++b) {
        // 1. Mean
        double sum = 0;
        for(int i=0; i<emb; ++i) sum += data[b*emb+i];
        double mean = sum / emb;
        (*cache_mean)[b] = mean;
        
        // 2. Variance
        double sum_sq = 0;
        for(int i=0; i<emb; ++i) {
            double d = data[b*emb+i] - mean;
            sum_sq += d*d;
        }
        double var = sum_sq / emb;
        double rstd = 1.0 / std::sqrt(var + eps);
        (*cache_rstd)[b] = rstd;
        
        // 3. Normalize & Scale
        for(int i=0; i<emb; ++i) {
            double n = (data[b*emb+i] - mean) * rstd;
            double g = gamma->data[i];
            double bt = beta->data[i];
            out->data[b*emb+i] = n * g + bt;
        }
    }
    
    out->_prev = {shared_from_this(), gamma, beta};
    out->_backward = [out, this_ptr=shared_from_this(), gamma, beta, cache_mean, cache_rstd, batch, emb]() {
        for(int b=0; b<batch; ++b) {
            double mean = (*cache_mean)[b];
            double rstd = (*cache_rstd)[b];
            
            double d_sigma = 0;
            double d_mean = 0;
            
            for(int i=0; i<emb; ++i) {
                double g = gamma->data[i];
                double dy = out->grad[b*emb+i];
                double x_hat = (this_ptr->data[b*emb+i] - mean) * rstd;
                
                // dL/dGamma
                gamma->grad[i] += dy * x_hat;
                // dL/dBeta
                beta->grad[i] += dy;
                
                // Backprop through scaling
                double dx_hat = dy * g;
                
                // Accumulate for mean/std
                d_sigma += dx_hat * (this_ptr->data[b*emb+i] - mean);
                d_mean -= dx_hat;
            }
            
            d_sigma *= -0.5 * rstd * rstd * rstd;
            
            // Final gradients w.r.t input
            for(int i=0; i<emb; ++i) {
                double dy = out->grad[b*emb+i];
                double g = gamma->data[i];
                double dx_hat = dy * g;
                
                double term1 = dx_hat * rstd;
                double term2 = d_sigma * 2 * (this_ptr->data[b*emb+i] - mean) / emb;
                double term3 = d_mean * rstd / emb;
                
                this_ptr->grad[b*emb+i] += term1 + term2 + (d_mean + d_sigma * 0) / emb; 
            }
        }
    };
    return out;
}

TensorPtr Tensor::tanh() {
    auto out = std::make_shared<Tensor>(shape);
    for(int i=0; i<size(); ++i) out->data[i] = std::tanh(data[i]);
    out->_prev = {shared_from_this()};
    out->_backward = [out, this_ptr=shared_from_this()]() {
        for(int i=0; i<out->size(); ++i) {
            double t = out->data[i];
            this_ptr->grad[i] += (1.0 - t * t) * out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::cross_entropy_loss(const std::vector<int>& targets) {
    int batch = shape[0], classes = shape[1];
    auto out = std::make_shared<Tensor>(std::vector<int>{1,1});
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

TensorPtr Tensor::mul(TensorPtr other) {
    auto out = std::make_shared<Tensor>(shape);
    bool broadcast = (other->size() < size());
    int mod = other->size();

    #pragma omp parallel for
    for (int i = 0; i < size(); ++i) {
        double b_val = broadcast ? other->data[i % mod] : other->data[i];
        out->data[i] = data[i] * b_val;
    }

    out->_prev = {shared_from_this(), other};
    out->_backward = [out, this_ptr=shared_from_this(), other, broadcast, mod]() {
        for (int i = 0; i < out->size(); ++i) {
            double g = out->grad[i];
            double b_val = broadcast ? other->data[i % mod] : other->data[i];
            
            this_ptr->grad[i] += b_val * g; // dL/da = b * g
            
            if (broadcast) {
                other->grad[i % mod] += this_ptr->data[i] * g; // dL/db = a * g
            } else {
                other->grad[i] += this_ptr->data[i] * g;
            }
        }
    };
    return out;
}

TensorPtr Tensor::relu() {
    auto out = std::make_shared<Tensor>(shape);
    
    #pragma omp parallel for
    for(int i=0; i<size(); ++i) {
        out->data[i] = (data[i] > 0) ? data[i] : 0.0;
    }
    
    out->_prev = {shared_from_this()};
    out->_backward = [out, this_ptr=shared_from_this()]() {
        for(int i=0; i<out->size(); ++i) {
            if (out->data[i] > 0) {
                this_ptr->grad[i] += out->grad[i];
            }
        }
    };
    return out;
}

TensorPtr Tensor::exp() {
    auto out = std::make_shared<Tensor>(shape);
    #pragma omp parallel for
    for(int i=0; i<size(); ++i) out->data[i] = std::exp(data[i]);
    
    out->_prev = {shared_from_this()};
    out->_backward = [out, this_ptr=shared_from_this()]() {
        for(int i=0; i<out->size(); ++i) {
            this_ptr->grad[i] += out->data[i] * out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::log() {
    auto out = std::make_shared<Tensor>(shape);
    #pragma omp parallel for
    for(int i=0; i<size(); ++i) out->data[i] = std::log(data[i]);
    
    out->_prev = {shared_from_this()};
    out->_backward = [out, this_ptr=shared_from_this()]() {
        for(int i=0; i<out->size(); ++i) {
            this_ptr->grad[i] += (1.0 / this_ptr->data[i]) * out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::sum() {
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1});
    double total = 0;
    
    for(double v : data) total += v;
    out->data[0] = total;
    
    out->_prev = {shared_from_this()};
    out->_backward = [out, this_ptr=shared_from_this()]() {
        double g = out->grad[0];
        for(int i=0; i<this_ptr->size(); ++i) {
            this_ptr->grad[i] += g;
        }
    };
    return out;
}

TensorPtr Tensor::max() {
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1});
    double max_val = -1e18;
    int idx = 0;
    
    for(int i=0; i<size(); ++i) {
        if(data[i] > max_val) {
            max_val = data[i];
            idx = i;
        }
    }
    out->data[0] = max_val;
    
    out->_prev = {shared_from_this()};
    out->_backward = [out, this_ptr=shared_from_this(), idx]() {
        this_ptr->grad[idx] += out->grad[0];
    };
    return out;
}

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
std::string Tensor::to_string() { 
    std::string s = "Tensor(";
    for(int i=0; i<shape.size(); ++i) s += (i>0?"x":"") + std::to_string(shape[i]);
    return s + ")"; 
}
