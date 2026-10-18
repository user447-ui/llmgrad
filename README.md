# 🔥 llmgrad

![Build Status](https://img.shields.io/github/actions/workflow/status/user447-ui/llmgrad/tests.yml?label=tests)
![PyPI](https://img.shields.io/pypi/v/llmgrad)
![License](https://img.shields.io/badge/license-MIT-blue)

**llmgrad** is a lightweight Deep Learning framework written in **C++** with **Python bindings**.
It implements a dynamic computation graph (Reverse-Mode Autodiff) with a C++ backend and OpenMP parallelization.

The goal of this project is to make modern deep-learning framework internals approachable by building the tensor and autograd core from scratch.

---

## 🚀 Key Features

* **C++ tensor core:** matrix multiplication, elementwise operations, activations and reductions.
* **Reverse-mode autograd:** dynamic computation graphs and gradient accumulation on leaf tensors.
* **Neural-network API:** `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Sequential`.
* **NLP primitives:** embedding lookup and context-window embedding concatenation.
* **Serialization:** Safetensors support for model parameters.
* **Cross-entropy:** numerically stable logits-based cross-entropy loss.

## 📥 Installation

Install from PyPI:

```bash
python -m pip install llmgrad
```

Build from source:

```bash
git clone https://github.com/user447-ui/llmgrad.git
cd llmgrad
python -m pip install .
```

## 🧪 Tests

The test suite checks numerical gradients against finite differences, shape and target validation, repeated backward calls, and Python module/optimizer integration.

```bash
python -m pip install -e . pytest numpy
python -m pytest -q
```

The CI workflow runs this suite on pushes to `main` and release branches, and on pull requests.

## 💻 Usage Example

```python
import ctensor
import llmgrad.nn as nn
import llmgrad.optim as optim

model = nn.Sequential([
    nn.Linear(4, 16),
    nn.Tanh(),
    nn.LayerNorm(16),
    nn.Linear(16, 3),
])

x = ctensor.Tensor.from_list([2, 4], [0.1, 0.2, 0.3, 0.4,
                                      -0.1, 0.5, 0.2, 0.7])
targets = [0, 2]
optimizer = optim.SGD(model.parameters(), lr=0.05)

logits = model(x)
loss = logits.cross_entropy(targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("loss:", loss.data[0])
```

## Notes

* `backward()` is defined for scalar outputs. Call a reduction such as `sum()` or use a scalar loss before calling it.
* Elementwise `add` and `mul` support equal shapes or row-broadcasting of a right-hand operand shaped `[1, columns]`.
* `matmul` currently supports 2D tensors only.
