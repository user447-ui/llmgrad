import pytest
import ctensor


def tensor(shape, values):
    return ctensor.Tensor.from_list(shape, values)


def scalar_value(t):
    assert len(t.data) == 1
    return t.data[0]


def finite_difference(values, loss_fn, index, eps=1e-6):
    plus = list(values)
    minus = list(values)
    plus[index] += eps
    minus[index] -= eps
    return (loss_fn(plus) - loss_fn(minus)) / (2.0 * eps)


def test_matmul_gradients_match_finite_differences():
    a_values = [0.2, -0.4, 0.7, 0.3]
    b_values = [0.5, -0.1, 0.8, 0.6]
    a = tensor([2, 2], a_values)
    b = tensor([2, 2], b_values)
    loss = a.matmul(b).sum()
    loss.backward()

    def loss_a(values):
        return scalar_value(tensor([2, 2], values).matmul(tensor([2, 2], b_values)).sum())

    def loss_b(values):
        return scalar_value(tensor([2, 2], a_values).matmul(tensor([2, 2], values)).sum())

    for i in range(len(a_values)):
        assert a.grad[i] == pytest.approx(finite_difference(a_values, loss_a, i), abs=1e-6)
        assert b.grad[i] == pytest.approx(finite_difference(b_values, loss_b, i), abs=1e-6)


def test_layernorm_input_and_parameter_gradients_match_finite_differences():
    x_values = [0.2, -0.7, 1.1, 0.4, 0.8, -0.3]
    gamma_values = [1.2, 0.7, -0.5]
    beta_values = [0.1, -0.2, 0.3]
    x = tensor([2, 3], x_values)
    gamma = tensor([3], gamma_values)
    beta = tensor([3], beta_values)
    x.layernorm(gamma, beta).sum().backward()

    def loss_x(values):
        return scalar_value(tensor([2, 3], values).layernorm(
            tensor([3], gamma_values), tensor([3], beta_values)).sum())

    def loss_gamma(values):
        return scalar_value(tensor([2, 3], x_values).layernorm(
            tensor([3], values), tensor([3], beta_values)).sum())

    def loss_beta(values):
        return scalar_value(tensor([2, 3], x_values).layernorm(
            tensor([3], gamma_values), tensor([3], values)).sum())

    for i in range(len(x_values)):
        assert x.grad[i] == pytest.approx(finite_difference(x_values, loss_x, i), abs=2e-5)
    for i in range(3):
        assert gamma.grad[i] == pytest.approx(finite_difference(gamma_values, loss_gamma, i), abs=2e-5)
        assert beta.grad[i] == pytest.approx(finite_difference(beta_values, loss_beta, i), abs=2e-5)


def test_cross_entropy_gradients_match_finite_differences():
    values = [1.2, -0.3, 0.1, -0.4, 0.6, 0.2]
    targets = [0, 2]
    logits = tensor([2, 3], values)
    logits.cross_entropy(targets).backward()

    def loss_fn(candidate):
        return scalar_value(tensor([2, 3], candidate).cross_entropy(targets))

    for i in range(len(values)):
        assert logits.grad[i] == pytest.approx(finite_difference(values, loss_fn, i), abs=2e-6)


def test_repeated_backward_accumulates_leaf_grads_without_reusing_intermediate_grads():
    x = tensor([1, 2], [0.2, -0.4])
    loss = x.tanh().sum()
    loss.backward()
    once = list(x.grad)
    loss.backward()
    for actual, expected in zip(x.grad, once):
        assert actual == pytest.approx(2.0 * expected, abs=1e-12)


def test_backward_requires_scalar_output():
    x = tensor([1, 2], [1.0, 2.0])
    with pytest.raises(ValueError, match="scalar"):
        x.backward()
