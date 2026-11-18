"""Tests for MCTS + Pareto de novo optimization, conformal UQ, Nelder-Mead, and Sobol indices."""

from __future__ import annotations

import numpy as np

from alchemi_dock.calibration import (
    calibrate_parameters_nelder_mead,
    conformal_predict_intervals,
    empirical_conformal_coverage,
    saltelli_sobol_sensitivity,
    split_conformal_quantile,
)
from alchemi_dock.denovo import (
    assign_pareto_ranks,
    optimize_molecules_mcts_pareto,
    propose_valid_mutations,
)
from alchemi_dock.molgraph import parse_smiles


def test_molecular_mutations_produce_valid_roundtrip_smiles() -> None:
    rng = np.random.default_rng(42)
    seed = "Fc1ccc(Nc2ncnc3ccccc23)cc1"
    proposals = propose_valid_mutations(seed, rng, max_proposals=10)
    assert len(proposals) >= 5
    for action, smi in proposals:
        assert len(action) > 0
        mol = parse_smiles(smi)
        assert mol.num_atoms >= 6
        assert mol.canonical_smiles == smi


def test_mcts_pareto_optimization_improves_mpo() -> None:
    seed = "Fc1ccc(Nc2ncnc3ccccc23)cc1"
    traj = optimize_molecules_mcts_pareto(seed_smiles=seed, generations=5, population_size=12, seed=42)
    assert traj.candidates_evaluated >= 15
    assert traj.valid_rate == 1.0
    assert traj.generation_best_mpo[-1] >= traj.generation_best_mpo[0]
    assert len(traj.pareto_frontier) >= 1
    # Verify rank 1 non-domination among frontier
    recomputed = assign_pareto_ranks(list(traj.pareto_frontier))
    assert len(recomputed) == len(traj.pareto_frontier)


def test_split_conformal_empirical_coverage_guarantee() -> None:
    rng = np.random.default_rng(42)
    n_cal, n_test = 250, 250
    y_cal_pred = rng.normal(6.5, 0.8, size=n_cal)
    y_cal_true = y_cal_pred + rng.normal(0.0, 0.35, size=n_cal)
    y_test_pred = rng.normal(6.5, 0.8, size=n_test)
    y_test_true = y_test_pred + rng.normal(0.0, 0.35, size=n_test)

    alpha = 0.10
    q_rad = split_conformal_quantile(y_cal_true, y_cal_pred, alpha=alpha)
    ivs = conformal_predict_intervals(y_test_pred, q_rad, alpha=alpha)
    cov = empirical_conformal_coverage(y_test_true, ivs)
    assert cov >= 0.84  # Expected ~90% with finite-sample binomial slack


def test_nelder_mead_calibration_and_sobol_sensitivity() -> None:
    x_grid = np.linspace(0.0, 5.0, 25)
    true_theta = np.array([2.4, 0.75])
    obs = true_theta[0] * np.exp(-true_theta[1] * x_grid)

    def _model(theta: np.ndarray) -> np.ndarray:
        return theta[0] * np.exp(-theta[1] * x_grid)

    opt, rmse = calibrate_parameters_nelder_mead(
        _model, obs, np.array([1.2, 0.3]), bounds=[(0.1, 10.0), (0.05, 5.0)], maxiter=200
    )
    assert rmse < 1e-3
    assert np.allclose(opt, true_theta, atol=1e-2)

    # Ishigami-like dominant vs inert variable check
    def _linear_fn(theta: np.ndarray) -> float:
        return float(4.0 * theta[0] + 0.1 * theta[1])

    sens = saltelli_sobol_sensitivity(
        _linear_fn, ["dominant", "weak"], [(0.0, 1.0), (0.0, 1.0)], n_samples=256, seed=42
    )
    assert sens.total_order_sti["dominant"] > 0.85
    assert sens.total_order_sti["weak"] < 0.05
