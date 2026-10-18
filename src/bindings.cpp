#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "tensor.hpp"

namespace py = pybind11;

TensorPtr embedding_forward(TensorPtr w, const std::vector<int>& idx) {
    if (!w || w->shape.size() != 2) {
        throw std::invalid_argument("Embedding: weight must be a 2D tensor");
    }
    const int rows = static_cast<int>(idx.size());
    const int cols = w->shape[1];
    for (int index : idx) {
        if (index < 0 || index >= w->shape[0]) {
            throw std::out_of_range("Embedding: index is out of range");
        }
    }
    auto out = Tensor::zeros(rows > 0 ? rows : 1, cols);
    if (rows == 0) {
        out = std::make_shared<Tensor>(std::vector<int>{1, cols});
        out->data.clear();
        out->grad.clear();
        out->shape = {1, cols};
    } else {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                out->data[i * cols + j] = w->data[idx[i] * cols + j];
    }

    out->_prev = {w};
    out->_backward = [out, w, idx, cols]() {
        for (int i = 0; i < static_cast<int>(idx.size()); ++i)
            for (int j = 0; j < cols; ++j)
                w->grad[idx[i] * cols + j] += out->grad[i * cols + j];
    };
    return out;
}

TensorPtr embedding_concat(TensorPtr w, const std::vector<int>& flat_indices, int block_size) {
    if (!w || w->shape.size() != 2) {
        throw std::invalid_argument("Embedding: weight must be a 2D tensor");
    }
    if (block_size <= 0) throw std::invalid_argument("Embedding: block_size must be positive");
    const int total_indices = static_cast<int>(flat_indices.size());
    if (total_indices % block_size != 0) {
        throw std::invalid_argument("Embedding: index count must be divisible by block_size");
    }
    const int batch_size = total_indices / block_size;
    const int emb_dim = w->shape[1];
    for (int index : flat_indices) {
        if (index < 0 || index >= w->shape[0]) {
            throw std::out_of_range("Embedding: index is out of range");
        }
    }
    const int out_cols = emb_dim * block_size;
    auto out = std::make_shared<Tensor>(std::vector<int>{batch_size > 0 ? batch_size : 1, out_cols});
    if (batch_size == 0) return out;

    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < block_size; ++k) {
            const int token_idx = flat_indices[b * block_size + k];
            for (int d = 0; d < emb_dim; ++d) {
                out->data[b * out_cols + k * emb_dim + d] = w->data[token_idx * emb_dim + d];
            }
        }
    }

    out->_prev = {w};
    out->_backward = [out, w, flat_indices, block_size, batch_size, emb_dim, out_cols]() {
        for (int b = 0; b < batch_size; ++b) {
            for (int k = 0; k < block_size; ++k) {
                const int token_idx = flat_indices[b * block_size + k];
                for (int d = 0; d < emb_dim; ++d) {
                    w->grad[token_idx * emb_dim + d] += out->grad[b * out_cols + k * emb_dim + d];
                }
            }
        }
    };
    return out;
}

PYBIND11_MODULE(ctensor, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def_static("zeros", &Tensor::zeros)
        .def_static("randn", &Tensor::randn)
        .def_static("from_list", &Tensor::from_list)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("shape", &Tensor::shape)
        .def("matmul", &Tensor::matmul)
        .def("add", &Tensor::add)
        .def("mul", &Tensor::mul)
        .def("tanh", &Tensor::tanh)
        .def("relu", &Tensor::relu)
        .def("exp", &Tensor::exp)
        .def("log", &Tensor::log)
        .def("sum", &Tensor::sum)
        .def("max", &Tensor::max)
        .def("reshape", &Tensor::reshape)
        .def("layernorm", &Tensor::layernorm, py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5)
        .def("backward", &Tensor::backward)
        .def("cross_entropy", &Tensor::cross_entropy_loss)
        .def("__add__", &Tensor::add)
        .def("__mul__", &Tensor::mul)
        .def("__matmul__", &Tensor::matmul)
        .def("__repr__", &Tensor::to_string)
        .def("zero_grad", [](TensorPtr t) { std::fill(t->grad.begin(), t->grad.end(), 0.0); })
        .def("sgd_step", [](TensorPtr t, double lr) {
            if (t->grad.size() != t->data.size()) throw std::runtime_error("Gradient/data size mismatch");
            for (size_t i = 0; i < t->data.size(); ++i) t->data[i] -= lr * t->grad[i];
        });

    m.def("embedding_forward", &embedding_forward);
    m.def("embedding_concat", &embedding_concat);
}
