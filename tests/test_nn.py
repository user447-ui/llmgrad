import pytest

import ctensor
import llmgrad.nn as nn
import llmgrad.optim as optim


def test_linear_layer_backward_and_sgd_step():
    layer = nn.Linear(2, 1)
    x = ctensor.Tensor.from_list([2, 2], [1.0, 2.0, -1.0, 0.5])
    before = list(layer.weight.data)
    layer(x).sum().backward()

    assert any(abs(g) > 0 for g in layer.weight.grad)
    optimizer = optim.SGD(layer.parameters(), lr=0.05)
    optimizer.step()
    assert layer.weight.data != before


def test_layernorm_module_runs_backward():
    layer = nn.LayerNorm(3)
    x = ctensor.Tensor.from_list([2, 3], [0.1, 0.4, -0.2, 0.8, -0.3, 0.5])
    layer(x).sum().backward()
    assert len(layer.gamma.grad) == 3
    assert len(layer.beta.grad) == 3
    assert layer.beta.grad == pytest.approx([2.0, 2.0, 2.0])
