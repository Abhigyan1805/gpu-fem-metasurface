"""The assembly runs on any torch device; CUDA results must equal CPU results."""

import numpy as np
import pytest
import torch

from fem.assembly import assemble
from fem.designs import get_design
from fem.mesh import build_mesh


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_assembly_matches_cpu():
    mesh = build_mesh(get_design("class9"), "coarse")
    a = assemble(mesh, pec_metal=False, order=2, device="cpu")
    b = assemble(mesh, pec_metal=False, order=2, device="cuda")
    assert np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
    for k in a.parts:
        np.testing.assert_allclose(a.parts[k], b.parts[k], rtol=1e-10, atol=1e-14)


def test_cpu_orders_share_conventions():
    mesh = build_mesh(get_design("class9"), "coarse")
    for order in (1, 2):
        s = assemble(mesh, pec_metal=False, order=order, device="cpu")
        assert s.indptr[-1] == len(s.indices)
        assert set(s.parts) >= {"S", "M_sub", "M_sup", "M_air", "B_port", "B_sheet"}
