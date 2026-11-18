"""Tests for vectorized pure-NumPy GNN architectures (GCN, MPNN, GATv2),
exact analytical gradients, training convergence, and scaffold-split guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest

from alchemi_dock.benchmark import load_kinase_benchmark_dataset
from alchemi_dock.descriptors import bemis_murcko_scaffold
from alchemi_dock.gnn import (
    ATOM_FEAT_DIM,
    MolecularGNN,
    featurize_mol_tensor,
    has_torch_adapter,
    scaffold_split_indices,
)
from alchemi_dock.molgraph import parse_smiles


@pytest.mark.parametrize("arch", ["gcn", "mpnn", "gatv2"])
def test_gnn_analytical_gradients_match_finite_difference(arch: str) -> None:
    mol = parse_smiles("COc1ccc(Nc2ncnc3ccccc23)cc1")
    graph = featurize_mol_tensor(mol)
    assert graph.atom_features.shape == (mol.num_atoms, ATOM_FEAT_DIM)

    model = MolecularGNN(architecture=arch, hidden_dim=8, num_layers=2, num_heads=2, seed=7)
    pred, attn, cache = model.forward_single(graph)
    assert abs(float(np.sum(attn)) - 1.0) < 1e-9

    target = 7.25
    d_pred = 2.0 * (pred - target)
    grads = model.backward_single(d_pred, cache)

    eps = 1e-6
    rng = np.random.default_rng(123)
    for name, param in model.params.items():
        # Probe up to 6 entries per parameter tensor for speed & full layer coverage
        flat_p = param.ravel()
        flat_g = grads[name].ravel()
        indices = rng.choice(len(flat_p), size=min(6, len(flat_p)), replace=False)
        for idx in indices:
            orig = flat_p[idx]
            flat_p[idx] = orig + eps
            p_plus, _, _ = model.forward_single(graph)
            l_plus = (p_plus - target) ** 2

            flat_p[idx] = orig - eps
            p_minus, _, _ = model.forward_single(graph)
            l_minus = (p_minus - target) ** 2
            flat_p[idx] = orig

            fd = (l_plus - l_minus) / (2.0 * eps)
            ana = float(flat_g[idx])
            assert abs(ana - fd) <= 1e-4 + 1e-3 * max(abs(ana), abs(fd)), (
                f"{arch} param={name}[{idx}]: ana={ana} vs fd={fd}"
            )


def test_scaffold_split_has_zero_scaffold_leakage() -> None:
    _, smiles_list, _ = load_kinase_benchmark_dataset(seed=42)
    tr_idx, va_idx = scaffold_split_indices(smiles_list, val_fraction=0.30)
    assert set(tr_idx).isdisjoint(set(va_idx))
    assert len(tr_idx) + len(va_idx) == len(smiles_list)

    tr_scaffs = {bemis_murcko_scaffold(parse_smiles(smiles_list[i])).murcko_smiles for i in tr_idx}
    va_scaffs = {bemis_murcko_scaffold(parse_smiles(smiles_list[i])).murcko_smiles for i in va_idx}
    # Non-empty Murcko scaffolds should not leak across split
    tr_nonempty = {s for s in tr_scaffs if s}
    va_nonempty = {s for s in va_scaffs if s}
    assert tr_nonempty.isdisjoint(va_nonempty)


@pytest.mark.parametrize("arch", ["gcn", "mpnn", "gatv2"])
def test_gnn_training_converges_and_learns_sar(arch: str) -> None:
    _, smiles_list, targets = load_kinase_benchmark_dataset(seed=42)
    graphs = [featurize_mol_tensor(parse_smiles(s)) for s in smiles_list]
    tr_idx, va_idx = scaffold_split_indices(smiles_list, val_fraction=0.25)

    model = MolecularGNN(architecture=arch, hidden_dim=24, num_layers=2, num_heads=2, seed=42)
    report = model.fit(
        [graphs[i] for i in tr_idx],
        targets[tr_idx],
        [graphs[i] for i in va_idx],
        targets[va_idx],
        epochs=20,
        lr=1e-2,
    )
    assert report.train_losses[-1] < report.train_losses[0]
    assert report.train_metrics.rmse < 0.85
    assert isinstance(has_torch_adapter(), bool)
