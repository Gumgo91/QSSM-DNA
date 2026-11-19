import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def dev():
    return "cpu"  # the unit tests run on CPU


def random_density(b: int, q: int, seed: int = 0) -> torch.Tensor:
    """Random valid density matrices [B,D,D] complex128 (mixed states)."""
    g = torch.Generator().manual_seed(seed)
    d = 1 << q
    out = []
    for _ in range(b):
        a = torch.randn(d, d, generator=g, dtype=torch.complex128)
        r = a @ a.conj().T
        r = r / r.trace()
        out.append(r)
    return torch.stack(out)


def random_pure_density(q: int, seed: int = 1) -> np.ndarray:
    """Single random pure-state density matrix, numpy complex128."""
    rng = np.random.default_rng(seed)
    d = 1 << q
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())
