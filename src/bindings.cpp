#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.hpp"

namespace py = pybind11;

TensorPtr embedding_concat(TensorPtr w, const std::vector<int>& flat_indices, int block_size) {
    int total_indices = flat_indices.size();
    if (total_indices % block_size != 0) throw std::runtime_error("Indices size must be divisible by block_size");
    
    int batch_size = total_indices / block_size;
    int emb_dim = w->shape[1];
    int out_cols = emb_dim * block_size;
    
    auto out = Tensor::zeros(batch_size, out_cols);
    
    // Forward
    for(int b=0; b<batch_size; ++b) {
        for(int k=0; k<block_size; ++k) {
            int token_idx = flat_indices[b * block_size + k];
            for(int d=0; d<emb_dim; ++d) {
                out->data[b * out_cols + k * emb_dim + d] = w->data[token_idx * emb_dim + d];
            }
        }
    }
    
    out->_prev = {w};
    out->_backward = [out, w, flat_indices, block_size, batch_size, emb_dim, out_cols]() {
        for(int b=0; b<batch_size; ++b) {
            for(int k=0; k<block_size; ++k) {
                int token_idx = flat_indices[b * block_size + k];
                for(int d=0; d<emb_dim; ++d) {
                    w->grad[token_idx * emb_dim + d] += out->grad[b * out_cols + k * emb_dim + d];
                }
            }
        }
    };
    return out;
}

// --- MODULE ---

PYBIND11_MODULE(ctensor, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def_static("zeros", &Tensor::zeros)
        .def_static("randn", &Tensor::randn)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("shape", &Tensor::shape)
        .def("matmul", &Tensor::matmul)
        .def("add", &Tensor::add)
        .def("tanh", &Tensor::tanh)
        .def("backward", &Tensor::backward)
        .def("cross_entropy", &Tensor::cross_entropy_loss)
        .def("__add__", &Tensor::add)
        .def("__matmul__", &Tensor::matmul)
        .def("__repr__", &Tensor::to_string)
        .def("zero_grad", [](TensorPtr t) { std::fill(t->grad.begin(), t->grad.end(), 0.0); })
        .def("sgd_step", [](TensorPtr t, double lr) {
             for(size_t i=0; i<t->data.size(); ++i) t->data[i] -= lr * t->grad[i];
        });

    m.def("embedding_concat", &embedding_concat);
}
