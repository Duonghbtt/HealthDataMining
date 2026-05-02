from __future__ import annotations

import random

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
torch = pytest.importorskip("torch")

from src.training.train_core import set_seed


def test_set_seed_is_reproducible_across_python_numpy_and_torch() -> None:
    set_seed(123)
    first_python = random.random()
    first_numpy = float(np.random.rand())
    first_torch = torch.rand(4)

    set_seed(123)
    second_python = random.random()
    second_numpy = float(np.random.rand())
    second_torch = torch.rand(4)

    assert first_python == pytest.approx(second_python)
    assert first_numpy == pytest.approx(second_numpy)
    assert torch.allclose(first_torch, second_torch)
