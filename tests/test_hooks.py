"""Tests for ActivationCapture forward hooks."""

import pytest
import torch
import torch.nn as nn

from ndt.core.hooks import ActivationCapture


@pytest.fixture
def model():
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))


def test_default_names_and_capture(model):
    capture = ActivationCapture()
    capture.register_hooks(model, [model[0], model[2]])
    x = torch.randn(3, 4)
    out = model(x)

    assert set(capture.activations) == {"Linear_0", "Linear_1"}
    torch.testing.assert_close(capture.get_activation("Linear_1"), out.detach())
    assert not capture.get_activation("Linear_0").requires_grad
    assert capture.get_activation("missing") is None
    capture.remove_hooks()


def test_mismatched_names_rejected(model):
    capture = ActivationCapture()
    with pytest.raises(ValueError, match=r"Number of layer names \(1\) must match .* \(2\)"):
        capture.register_hooks(model, [model[0], model[2]], ["only_one"])
    assert capture.hooks == []


def test_get_all_activations_returns_copy(model):
    capture = ActivationCapture()
    capture.register_hooks(model, [model[0]], ["first"])
    model(torch.randn(2, 4))

    snapshot = capture.get_all_activations()
    snapshot.pop("first")
    assert "first" in capture.activations

    capture.clear_activations()
    assert capture.activations == {}
    assert len(capture.hooks) == 1  # hooks survive clear_activations
    capture.remove_hooks()


def test_context_manager_removes_hooks(model):
    with ActivationCapture() as capture:
        capture.register_hooks(model, [model[0], model[2]], ["a", "b"])
        assert len(model[0]._forward_hooks) == 1
        model(torch.randn(2, 4))
        assert set(capture.activations) == {"a", "b"}

    assert len(model[0]._forward_hooks) == 0
    assert len(model[2]._forward_hooks) == 0
    assert capture.activations == {}
    assert capture._layer_names == {}

    # After removal, forward passes are no longer captured.
    model(torch.randn(2, 4))
    assert capture.activations == {}
