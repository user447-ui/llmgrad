import pytest
import ctensor


def tensor(shape, values):
    return ctensor.Tensor.from_list(shape, values)


def test_matmul_rejects_incompatible_inner_dimensions():
    left = tensor([2, 3], [1, 2, 3, 4, 5, 6])
    right = tensor([2, 2], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="inner dimensions"):
        left.matmul(right)


def test_add_rejects_incompatible_shapes():
    left = tensor([2, 3], [1, 2, 3, 4, 5, 6])
    right = tensor([2, 2], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="shapes must match"):
        left.add(right)


def test_row_broadcast_add_and_gradient():
    x = tensor([2, 3], [1, 2, 3, 4, 5, 6])
    bias = tensor([1, 3], [0.1, 0.2, 0.3])
    x.add(bias).sum().backward()
    assert x.grad == pytest.approx([1.0] * 6)
    assert bias.grad == pytest.approx([2.0, 2.0, 2.0])


def test_cross_entropy_rejects_wrong_target_count_and_range():
    logits = tensor([2, 3], [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="target count"):
        logits.cross_entropy([1])
    with pytest.raises(IndexError, match="out of range"):
        logits.cross_entropy([0, 3])


def test_layernorm_rejects_wrong_parameter_size():
    x = tensor([2, 3], [1, 2, 3, 4, 5, 6])
    gamma = tensor([2], [1, 1])
    beta = tensor([3], [0, 0, 0])
    with pytest.raises(ValueError, match="one value per feature"):
        x.layernorm(gamma, beta)


def test_tensor_rejects_zero_dimensions():
    with pytest.raises(ValueError, match="positive"):
        ctensor.Tensor.from_list([0, 2], [])
