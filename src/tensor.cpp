#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace {
int checked_size(const std::vector<int>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape must not be empty");
    }
    long long total = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive");
        }
        total *= dim;
        if (total > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Tensor is too large");
        }
    }
    return static_cast<int>(total);
}

bool same_shape(const std::vector<int>& a, const std::vector<int>& b) {
    return a == b;
}

bool row_broadcast(const Tensor& lhs, const Tensor& rhs) {
    return lhs.shape.size() == 2 && rhs.shape.size() == 2 &&
           rhs.shape[0] == 1 && rhs.shape[1] == lhs.shape[1];
}

void validate_binary_shapes(const Tensor& lhs, const Tensor& rhs,
                            const char* operation) {
    if (!same_shape(lhs.shape, rhs.shape) && !row_broadcast(lhs, rhs)) {
        throw std::invalid_argument(std::string(operation) +
            ": shapes must match or the right operand must have shape [1, columns]");
    }
}
}  // namespace

int Tensor::size() const {
    return checked_size(shape);
}

Tensor::Tensor(std::vector<int> s, std::vector<double> d) : shape(std::move(s)) {
    const int n = checked_size(shape);
    if (d.empty()) {
        data.resize(static_cast<size_t>(n), 0.0);
    } else {
        if (static_cast<int>(d.size()) != n) {
            throw std::invalid_argument("Data size mismatch in constructor");
        }
        data = std::move(d);
    }
    grad.resize(static_cast<size_t>(n), 0.0);
    _backward = [] {};
}

TensorPtr Tensor::zeros(int rows, int cols) {
    return std::make_shared<Tensor>(std::vector<int>{rows, cols});
}

TensorPtr Tensor::randn(int rows, int cols) {
    auto t = zeros(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(static_cast<double>(cols)));
    for (auto& v : t->data) v = dist(gen);
    return t;
}

TensorPtr Tensor::from_list(std::vector<int> shape, const std::vector<double>& list) {
    return std::make_shared<Tensor>(std::move(shape), list);
}

TensorPtr Tensor::reshape(const std::vector<int>& new_shape) {
    if (checked_size(new_shape) != size()) {
        throw std::invalid_argument("Reshape: total elements mismatch");
    }
    auto out = std::make_shared<Tensor>(new_shape, data);
    auto self = shared_from_this();
    out->_prev = {self};
    out->_backward = [out, self]() {
        for (int i = 0; i < out->size(); ++i) self->grad[i] += out->grad[i];
    };
    return out;
}

TensorPtr Tensor::add(TensorPtr other) {
    if (!other) throw std::invalid_argument("Add: other tensor must not be null");
    validate_binary_shapes(*this, *other, "Add");
    const bool broadcast = !same_shape(shape, other->shape);
    const int cols = shape.back();
    auto out = std::make_shared<Tensor>(shape);
    for (int i = 0; i < size(); ++i) {
        const int j = broadcast ? i % cols : i;
        out->data[i] = data[i] + other->data[j];
    }

    auto self = shared_from_this();
    out->_prev = {self, other};
    out->_backward = [out, self, other, broadcast, cols]() {
        for (int i = 0; i < out->size(); ++i) {
            self->grad[i] += out->grad[i];
            other->grad[broadcast ? i % cols : i] += out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::mul(TensorPtr other) {
    if (!other) throw std::invalid_argument("Mul: other tensor must not be null");
    validate_binary_shapes(*this, *other, "Mul");
    const bool broadcast = !same_shape(shape, other->shape);
    const int cols = shape.back();
    auto out = std::make_shared<Tensor>(shape);
    for (int i = 0; i < size(); ++i) {
        const int j = broadcast ? i % cols : i;
        out->data[i] = data[i] * other->data[j];
    }

    auto self = shared_from_this();
    out->_prev = {self, other};
    out->_backward = [out, self, other, broadcast, cols]() {
        for (int i = 0; i < out->size(); ++i) {
            const int j = broadcast ? i % cols : i;
            self->grad[i] += other->data[j] * out->grad[i];
            other->grad[j] += self->data[i] * out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::matmul(TensorPtr other) {
    if (!other) throw std::invalid_argument("Matmul: other tensor must not be null");
    if (shape.size() != 2 || other->shape.size() != 2) {
        throw std::invalid_argument("Matmul currently supports only 2D tensors");
    }
    const int m = shape[0], k = shape[1], n = other->shape[1];
    if (k != other->shape[0]) {
        throw std::invalid_argument("Matmul: inner dimensions must match");
    }

    auto out = zeros(m, n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) sum += data[i * k + p] * other->data[p * n + j];
            out->data[i * n + j] = sum;
        }
    }

    auto self = shared_from_this();
    out->_prev = {self, other};
    out->_backward = [out, self, other, m, k, n]() {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                const double g = out->grad[i * n + j];
                for (int p = 0; p < k; ++p) {
                    self->grad[i * k + p] += g * other->data[p * n + j];
                    other->grad[p * n + j] += self->data[i * k + p] * g;
                }
            }
        }
    };
    return out;
}

TensorPtr Tensor::layernorm(TensorPtr gamma, TensorPtr beta, double eps) {
    if (!gamma || !beta) throw std::invalid_argument("LayerNorm: gamma and beta must not be null");
    if (shape.size() != 2) throw std::invalid_argument("LayerNorm expects a 2D [batch, features] tensor");
    const int batch = shape[0], features = shape[1];
    if (gamma->size() != features || beta->size() != features) {
        throw std::invalid_argument("LayerNorm: gamma and beta must contain one value per feature");
    }
    if (eps <= 0.0 || !std::isfinite(eps)) {
        throw std::invalid_argument("LayerNorm: eps must be finite and positive");
    }

    auto out = std::make_shared<Tensor>(shape);
    auto mean_cache = std::make_shared<std::vector<double>>(batch, 0.0);
    auto rstd_cache = std::make_shared<std::vector<double>>(batch, 0.0);

    for (int b = 0; b < batch; ++b) {
        double mean = 0.0;
        for (int j = 0; j < features; ++j) mean += data[b * features + j];
        mean /= features;

        double variance = 0.0;
        for (int j = 0; j < features; ++j) {
            const double centered = data[b * features + j] - mean;
            variance += centered * centered;
        }
        variance /= features;
        const double rstd = 1.0 / std::sqrt(variance + eps);
        (*mean_cache)[b] = mean;
        (*rstd_cache)[b] = rstd;

        for (int j = 0; j < features; ++j) {
            const double xhat = (data[b * features + j] - mean) * rstd;
            out->data[b * features + j] = xhat * gamma->data[j] + beta->data[j];
        }
    }

    auto self = shared_from_this();
    out->_prev = {self, gamma, beta};
    out->_backward = [out, self, gamma, beta, mean_cache, rstd_cache, batch, features]() {
        for (int b = 0; b < batch; ++b) {
            const double mean = (*mean_cache)[b];
            const double rstd = (*rstd_cache)[b];
            double sum_dxhat = 0.0;
            double sum_dxhat_xhat = 0.0;

            for (int j = 0; j < features; ++j) {
                const int idx = b * features + j;
                const double xhat = (self->data[idx] - mean) * rstd;
                const double dy = out->grad[idx];
                const double dxhat = dy * gamma->data[j];
                gamma->grad[j] += dy * xhat;
                beta->grad[j] += dy;
                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * xhat;
            }

            // dx = rstd / N * (N*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
            for (int j = 0; j < features; ++j) {
                const int idx = b * features + j;
                const double xhat = (self->data[idx] - mean) * rstd;
                const double dxhat = out->grad[idx] * gamma->data[j];
                self->grad[idx] += (rstd / features) *
                    (features * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
            }
        }
    };
    return out;
}

TensorPtr Tensor::tanh() {
    auto out = std::make_shared<Tensor>(shape);
    for (int i = 0; i < size(); ++i) out->data[i] = std::tanh(data[i]);
    auto self = shared_from_this();
    out->_prev = {self};
    out->_backward = [out, self]() {
        for (int i = 0; i < out->size(); ++i) {
            const double t = out->data[i];
            self->grad[i] += (1.0 - t * t) * out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::exp() {
    auto out = std::make_shared<Tensor>(shape);
    #pragma omp parallel for
    for (int i = 0; i < size(); ++i) out->data[i] = std::exp(data[i]);
    auto self = shared_from_this();
    out->_prev = {self};
    out->_backward = [out, self]() {
        for (int i = 0; i < out->size(); ++i) self->grad[i] += out->data[i] * out->grad[i];
    };
    return out;
}

TensorPtr Tensor::log() {
    auto out = std::make_shared<Tensor>(shape);
    for (int i = 0; i < size(); ++i) {
        if (data[i] <= 0.0) throw std::domain_error("Log requires strictly positive inputs");
        out->data[i] = std::log(data[i]);
    }
    auto self = shared_from_this();
    out->_prev = {self};
    out->_backward = [out, self]() {
        for (int i = 0; i < out->size(); ++i) self->grad[i] += out->grad[i] / self->data[i];
    };
    return out;
}

TensorPtr Tensor::relu() {
    auto out = std::make_shared<Tensor>(shape);
    #pragma omp parallel for
    for (int i = 0; i < size(); ++i) out->data[i] = data[i] > 0.0 ? data[i] : 0.0;
    auto self = shared_from_this();
    out->_prev = {self};
    out->_backward = [out, self]() {
        for (int i = 0; i < out->size(); ++i) {
            if (self->data[i] > 0.0) self->grad[i] += out->grad[i];
        }
    };
    return out;
}

TensorPtr Tensor::sum() {
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1});
    out->data[0] = std::accumulate(data.begin(), data.end(), 0.0);
    auto self = shared_from_this();
    out->_prev = {self};
    out->_backward = [out, self]() {
        for (int i = 0; i < self->size(); ++i) self->grad[i] += out->grad[0];
    };
    return out;
}

TensorPtr Tensor::max() {
    if (data.empty()) throw std::invalid_argument("Max requires a non-empty tensor");
    const auto it = std::max_element(data.begin(), data.end());
    const int index = static_cast<int>(std::distance(data.begin(), it));
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1}, {*it});
    auto self = shared_from_this();
    out->_prev = {self};
    out->_backward = [out, self, index]() { self->grad[index] += out->grad[0]; };
    return out;
}

TensorPtr Tensor::cross_entropy_loss(const std::vector<int>& targets) {
    if (shape.size() != 2) throw std::invalid_argument("CrossEntropy expects [batch, classes] logits");
    const int batch = shape[0], classes = shape[1];
    if (static_cast<int>(targets.size()) != batch) {
        throw std::invalid_argument("CrossEntropy: target count must equal batch size");
    }
    for (int target : targets) {
        if (target < 0 || target >= classes) {
            throw std::out_of_range("CrossEntropy: target index is out of range");
        }
    }

    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1});
    auto probs = std::make_shared<std::vector<double>>(data.size(), 0.0);
    double total_loss = 0.0;
    for (int b = 0; b < batch; ++b) {
        const auto begin = data.begin() + b * classes;
        const double max_logit = *std::max_element(begin, begin + classes);
        double exp_sum = 0.0;
        for (int c = 0; c < classes; ++c) {
            const double e = std::exp(data[b * classes + c] - max_logit);
            (*probs)[b * classes + c] = e;
            exp_sum += e;
        }
        for (int c = 0; c < classes; ++c) (*probs)[b * classes + c] /= exp_sum;
        total_loss += std::log(exp_sum) + max_logit - data[b * classes + targets[b]];
    }
    out->data[0] = total_loss / batch;
    auto self = shared_from_this();
    out->_prev = {self};
    out->_backward = [out, self, targets, probs, batch, classes]() {
        const double scale = out->grad[0] / batch;
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < classes; ++c) {
                self->grad[b * classes + c] +=
                    ((*probs)[b * classes + c] - (c == targets[b] ? 1.0 : 0.0)) * scale;
            }
        }
    };
    return out;
}

void Tensor::backward() {
    if (size() != 1) {
        throw std::invalid_argument("Backward requires a scalar tensor; reduce the output first");
    }

    std::vector<TensorPtr> topo;
    std::set<TensorPtr> visited;
    std::function<void(TensorPtr)> build = [&](TensorPtr node) {
        if (visited.insert(node).second) {
            for (const auto& parent : node->_prev) build(parent);
            topo.push_back(node);
        }
    };
    build(shared_from_this());

    // Intermediate gradients belong to this traversal; leaf gradients accumulate
    // across calls, as expected by gradient-accumulation training loops.
    for (const auto& node : topo) {
        if (!node->_prev.empty()) std::fill(node->grad.begin(), node->grad.end(), 0.0);
    }
    grad[0] = 1.0;
    std::reverse(topo.begin(), topo.end());
    for (const auto& node : topo) node->_backward();
}

std::string Tensor::to_string() {
    std::string result = "Tensor(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) result += "x";
        result += std::to_string(shape[i]);
    }
    return result + ")";
}
