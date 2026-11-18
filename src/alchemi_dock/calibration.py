"""Uncertainty quantification, parameter calibration, and Saltelli Sobol sensitivity analysis.

Implements:
- Distribution-free Split Conformal Prediction intervals with finite-sample quantile correction
- Nelder-Mead simplex calibration for pharmacokinetic / PROTAC kinetic parameters
- Saltelli (2010) Sobol first-order (S_i) and total-order (S_Ti) global sensitivity indices
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .types import ConformalInterval, FloatArray, SensitivityReport


def split_conformal_quantile(
    y_cal_true: FloatArray,
    y_cal_pred: FloatArray,
    alpha: float = 0.10,
) -> float:
    """Compute finite-sample split conformal quantile radius q_{1-alpha}.

    Guarantees P(Y_{test} in [y_hat - q, y_hat + q]) >= 1 - alpha under exchangeability.
    """
    residuals = np.abs(y_cal_true - y_cal_pred)
    n = len(residuals)
    if n == 0:
        return 0.0
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(residuals, level, method="higher" if hasattr(np, "quantile") else "linear"))


def conformal_predict_intervals(
    y_test_pred: FloatArray,
    quantile_radius: float,
    alpha: float = 0.10,
) -> List[ConformalInterval]:
    """Wrap scalar point predictions into distribution-free conformal intervals."""
    return [
        ConformalInterval(
            prediction=float(val),
            lower=float(val - quantile_radius),
            upper=float(val + quantile_radius),
            alpha=alpha,
            quantile_radius=quantile_radius,
        )
        for val in y_test_pred
    ]


def empirical_conformal_coverage(
    y_test_true: FloatArray,
    intervals: Sequence[ConformalInterval],
) -> float:
    """Compute realized empirical coverage fraction across test conformal intervals."""
    if not intervals or len(y_test_true) == 0:
        return 0.0
    hits = sum(
        1 for y, iv in zip(y_test_true, intervals) if (iv.lower - 1e-12) <= float(y) <= (iv.upper + 1e-12)
    )
    return float(hits / len(intervals))


def calibrate_parameters_nelder_mead(
    model_fn: Callable[[FloatArray], FloatArray],
    observed_y: FloatArray,
    initial_guess: FloatArray,
    bounds: Sequence[Tuple[float, float]],
    maxiter: int = 120,
) -> Tuple[FloatArray, float]:
    """Calibrate continuous kinetic/biophysical parameters using bounded Nelder-Mead."""
    lows = np.array([b[0] for b in bounds], dtype=np.float64)
    highs = np.array([b[1] for b in bounds], dtype=np.float64)

    def _objective(theta: FloatArray) -> float:
        clipped = np.clip(theta, lows, highs)
        penalty = float(1e4 * np.sum((theta - clipped) ** 2))
        pred = model_fn(clipped)
        mse = float(np.mean((pred - observed_y) ** 2))
        return mse + penalty

    res = minimize(_objective, initial_guess, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-4})
    opt_theta = np.clip(res.x, lows, highs)
    rmse = float(np.sqrt(np.mean((model_fn(opt_theta) - observed_y) ** 2)))
    return opt_theta, rmse


def saltelli_sobol_sensitivity(
    scalar_fn: Callable[[FloatArray], float],
    param_names: Sequence[str],
    bounds: Sequence[Tuple[float, float]],
    n_samples: int = 64,
    seed: int = 42,
) -> SensitivityReport:
    """Estimate Saltelli-Jansen Sobol first-order (S_i) and total-order (S_Ti) indices."""
    rng = np.random.default_rng(seed)
    k = len(param_names)
    lows = np.array([b[0] for b in bounds], dtype=np.float64)
    highs = np.array([b[1] for b in bounds], dtype=np.float64)

    def _sample_lhs(n: int) -> FloatArray:
        mat = np.zeros((n, k), dtype=np.float64)
        for j in range(k):
            perm = rng.permutation(n)
            u = (perm + rng.random(n)) / n
            mat[:, j] = lows[j] + u * (highs[j] - lows[j])
        return mat

    mat_a = _sample_lhs(n_samples)
    mat_b = _sample_lhs(n_samples)

    y_a = np.array([scalar_fn(row) for row in mat_a], dtype=np.float64)
    y_b = np.array([scalar_fn(row) for row in mat_b], dtype=np.float64)
    y_all = np.concatenate([y_a, y_b])
    var_y = float(np.var(y_all))
    if var_y < 1e-15:
        zeros = {name: 0.0 for name in param_names}
        return SensitivityReport(
            parameter_names=list(param_names),
            first_order_si=zeros,
            total_order_sti=zeros.copy(),
            output_mean=float(np.mean(y_all)),
            output_std=0.0,
        )

    first_order: Dict[str, float] = {}
    total_order: Dict[str, float] = {}

    for i, name in enumerate(param_names):
        mat_ab_i = mat_a.copy()
        mat_ab_i[:, i] = mat_b[:, i]
        y_ab_i = np.array([scalar_fn(row) for row in mat_ab_i], dtype=np.float64)

        # Saltelli (2010) first-order and Jansen (1999) total-order estimators
        s_i = float(np.mean(y_b * (y_ab_i - y_a)) / var_y)
        s_ti = float(0.5 * np.mean((y_a - y_ab_i) ** 2) / var_y)
        first_order[name] = float(max(0.0, min(1.0, s_i)))
        total_order[name] = float(max(0.0, min(1.0, s_ti)))

    return SensitivityReport(
        parameter_names=list(param_names),
        first_order_si=first_order,
        total_order_sti=total_order,
        output_mean=float(np.mean(y_all)),
        output_std=float(math.sqrt(var_y)),
    )
