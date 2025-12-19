# 🔥 llmgrad

![Build Status](https://img.shields.io/github/actions/workflow/status/user447-ui/llmgrad/build_wheels.yml?label=build)
![PyPI](https://img.shields.io/pypi/v/llmgrad)
![License](https://img.shields.io/badge/license-MIT-blue)

**llmgrad** is a lightweight Deep Learning framework written in **C++** with **Python bindings**.
It implements a dynamic computation graph (Reverse-Mode Autodiff) similar to PyTorch, but runs on a highly optimized C++ backend using OpenMP for parallelization.

The goal of this project is to demystify how modern DL frameworks work under the hood by building one from scratch, bridging low-level memory management with high-level Python APIs.

---

## 🚀 Key Features

*   **⚡ High Performance Core:** Tensor operations (`matmul`, `add`, `tanh`, etc.) are implemented in C++ and accelerated using **OpenMP** (multi-threading).
*   **🧠 Autograd Engine:** Automatic differentiation with support for dynamic graphs.
*   **🛠 PyTorch-like API:** Familiar interface with `nn.Module`, `nn.Linear`, `optim.SGD`, and `Sequential` containers.
*   **📚 NLP Optimized:** Specialized C++ kernels for efficient Embedding concatenation (`embedding_concat`) used in Context Windows.
*   **🔧 Modern Architecture:**
    *   **LayerNorm** implementation in C++.
    *   **Safetensors** support for saving/loading weights.
    *   **Fused CrossEntropyLoss** for numerical stability.
*   **📦 Zero-Dependency Install:** Distributed as binary wheels for Linux, Windows, and macOS.

---

## 📥 Installation

You can install the latest release directly from PyPI:

```
pip install llmgrad

Or build from source (requires a C++ compiler):
git clone https://github.com/user447-ui/llmgrad.git
cd llmgrad
pip install .
```

---

## 💻 Usage Example
The API is designed to be nearly identical to PyTorch.

1. Training a Model
```
import llmgrad
import llmgrad.nn as nn
import llmgrad.optim as optim

# 1. Define the Model
model = nn.Sequential([
    nn.Embedding(vocab_size=27, embedding_dim=10),
    nn.Linear(in_features=10*3, out_features=64), # Context window = 3
    nn.LayerNorm(64),
    nn.Tanh(),
    nn.Linear(64, 27)
])

# 2. Setup Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 3. Training Loop
# x_batch: list of integers (indices)
# y_batch: list of integers (targets)
logits = model(x_batch)

loss = logits.cross_entropy(y_batch)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.data[0]}")
```

2. Saving & Loading
We use the generic .safetensors format for weight serialization.

```
# Save weights
model.save_pretrained("model.safetensors")

# Load weights
model.load_pretrained("model.safetensors")
```
