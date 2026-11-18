"""PROTAC ternary complex cooperativity and target degradation kinetics simulator.

Implements:
- Exact Douglass-Miller three-body binding equilibrium solver (brentq over analytical
  quadratic protein/E3 mass-conservation reduction, accurate to < 1e-12 uM)
- Linker-geometry and rigidity cooperativity (alpha) estimator
- Hook-effect bell-shaped ternary complex profiling across log-dose grids
- RK45 proteasomal target degradation ODEs computing Dmax, DC50, and AOC
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .descriptors import compute_clogp_and_mr, compute_tpsa
from .molgraph import identify_rotatable_bonds, parse_smiles
from .types import (
    FloatArray,
    ProtacDegradationResult,
    ProtacEquilibriumCurve,
    ProtacSpec,
)


def estimate_protac_cooperativity(
    linker_smiles: str,
    warhead_smiles: str = "",
    e3_ligand_smiles: str = "",
) -> float:
    """Estimate ternary complex cooperativity alpha from linker topology and rigidity.

    Very short linkers (< 5 heavy atoms) incur steric POI-E3 clash (alpha < 1).
    Optimal 7-13 atom semi-rigid linkers (rings/amides/alkynes) yield positive
    cooperativity (alpha > 1), while overly long flexible PEG chains suffer entropic loss.
    """
    linker = parse_smiles(linker_smiles)
    n_heavy = linker.num_atoms
    n_rot = len(identify_rotatable_bonds(linker))
    n_rings = len(linker.sssr_rings)
    tpsa = compute_tpsa(linker)
    clogp, _ = compute_clogp_and_mr(linker)

    # Optimal effective length around 9-11 heavy atoms
    length_term = math.exp(-((n_heavy - 10.0) / 4.2) ** 2)
    steric_clash = 0.18 if n_heavy < 5 else (0.55 if n_heavy < 7 else 1.0)
    rigidity_bonus = 1.0 + 0.45 * min(2, n_rings) - 0.04 * max(0, n_rot - 6)
    polarity_penalty = 1.0 - 0.002 * max(0.0, tpsa - 95.0)

    # Optional warhead/E3 lipophilic interface synergy
    ppi_synergy = 1.0
    if warhead_smiles and e3_ligand_smiles:
        wh = parse_smiles(warhead_smiles)
        e3 = parse_smiles(e3_ligand_smiles)
        lp_w, _ = compute_clogp_and_mr(wh)
        lp_e, _ = compute_clogp_and_mr(e3)
        ppi_synergy = 1.0 + 0.06 * max(0.0, min(4.0, lp_w + lp_e - 2.0))

    alpha = (
        steric_clash
        * (0.6 + 6.5 * length_term)
        * max(0.4, rigidity_bonus)
        * max(0.5, polarity_penalty)
        * ppi_synergy
    )
    return float(max(0.08, min(25.0, alpha * (1.0 + 0.02 * max(-2.0, min(2.0, clogp))))))


def solve_ternary_equilibrium_point(
    l_total_uM: float,
    p_total_uM: float,
    e_total_uM: float,
    kd_p_uM: float,
    kd_e_uM: float,
    alpha: float,
) -> Tuple[float, float, float, float, float, float]:
    """Solve exact Douglass-Miller three-body mass-conservation equilibrium at fixed L_0.

    Returns (L_free, P_free, E_free, PL, EL, PLE) in micromolar (uM).
    """
    if l_total_uM <= 0.0:
        return 0.0, p_total_uM, e_total_uM, 0.0, 0.0, 0.0

    kd_p = max(1e-9, kd_p_uM)
    kd_e = max(1e-9, kd_e_uM)
    al = max(1e-6, alpha)

    def _pe_given_l(l_free: float) -> Tuple[float, float]:
        a = l_free / kd_p
        b = l_free / kd_e
        c = al * l_free / (kd_p * kd_e)
        if c < 1e-15:
            p_free = p_total_uM / (1.0 + a)
            e_free = e_total_uM / (1.0 + b)
            return p_free, e_free
        quad_a = c * (1.0 + a)
        quad_b = (1.0 + a) * (1.0 + b) + c * (e_total_uM - p_total_uM)
        quad_c = -p_total_uM * (1.0 + b)
        disc = max(0.0, quad_b * quad_b - 4.0 * quad_a * quad_c)
        p_free = (-quad_b + math.sqrt(disc)) / (2.0 * quad_a)
        e_free = e_total_uM / (1.0 + b + c * p_free)
        return p_free, e_free

    def _residual(l_free: float) -> float:
        p_free, e_free = _pe_given_l(l_free)
        pl = (p_free * l_free) / kd_p
        el = (e_free * l_free) / kd_e
        ple = (al * p_free * e_free * l_free) / (kd_p * kd_e)
        return (l_free + pl + el + ple) - l_total_uM

    l_free = float(brentq(_residual, 0.0, l_total_uM, xtol=1e-14, rtol=1e-13, maxiter=120))
    p_free, e_free = _pe_given_l(l_free)
    pl = (p_free * l_free) / kd_p
    el = (e_free * l_free) / kd_e
    ple = (al * p_free * e_free * l_free) / (kd_p * kd_e)
    return l_free, p_free, e_free, pl, el, ple


def compute_protac_equilibrium_curve(
    spec: ProtacSpec,
    conc_grid_uM: Optional[FloatArray] = None,
) -> ProtacEquilibriumCurve:
    """Compute ternary complex [PLE] dose-response curve and verify mass-balance conservation."""
    alpha = (
        spec.cooperativity_alpha
        if spec.cooperativity_alpha is not None
        else estimate_protac_cooperativity(spec.linker_smiles, spec.warhead_smiles, spec.e3_ligand_smiles)
    )
    if conc_grid_uM is None:
        conc_grid_uM = np.logspace(-4, 2.5, 65)

    n = len(conc_grid_uM)
    ple_arr = np.zeros(n, dtype=np.float64)
    pl_arr = np.zeros(n, dtype=np.float64)
    el_arr = np.zeros(n, dtype=np.float64)
    max_err = 0.0

    for i, l0 in enumerate(conc_grid_uM):
        lf, pf, ef, pl, el, ple = solve_ternary_equilibrium_point(
            float(l0), spec.poi_total_uM, spec.e3_total_uM, spec.kd_poi_uM, spec.kd_e3_uM, alpha
        )
        ple_arr[i] = ple
        pl_arr[i] = pl
        el_arr[i] = el
        err_p = abs((pf + pl + ple) - spec.poi_total_uM)
        err_e = abs((ef + el + ple) - spec.e3_total_uM)
        err_l = abs((lf + pl + el + ple) - float(l0))
        max_err = max(max_err, err_p, err_e, err_l)

    peak_idx = int(np.argmax(ple_arr))
    peak_ternary = float(ple_arr[peak_idx])
    hook_conc = float(conc_grid_uM[peak_idx])
    peak_frac = peak_ternary / max(1e-12, min(spec.poi_total_uM, spec.e3_total_uM))

    return ProtacEquilibriumCurve(
        spec=spec,
        estimated_alpha=alpha,
        conc_grid_uM=conc_grid_uM,
        ternary_ple_uM=ple_arr,
        binary_pl_uM=pl_arr,
        binary_el_uM=el_arr,
        peak_ternary_uM=peak_ternary,
        hook_conc_uM=hook_conc,
        peak_fraction_e3_bound=float(peak_frac),
        mass_balance_max_err=float(max_err),
    )


def simulate_protac_degradation(
    spec: ProtacSpec,
    dose_uM: Optional[float] = None,
    hours: float = 24.0,
) -> ProtacDegradationResult:
    """Simulate time-dependent POI proteasomal degradation via RK45 coupled ternary ODEs.

    At each time step, POI abundance P(t) determines quasi-steady-state ternary complex
    [PLE](P(t), L_0, E_0), driving catalytic ubiquitination + proteasomal degradation:
        dP/dt = k_syn - k_deg_basal * P - k_ub * [PLE](P, E_0, L_0)
    """
    curve = compute_protac_equilibrium_curve(spec)
    chosen_dose = dose_uM if dose_uM is not None else curve.hook_conc_uM
    alpha = curve.estimated_alpha

    # Steady-state baseline POI when L_0 = 0 is k_syn / k_deg_basal (match spec.poi_total_uM if aligned)
    p0 = spec.k_syn_uM_per_h / max(1e-6, spec.k_deg_basal_per_h)

    def _simulate_at_dose(l0: float, t_eval: FloatArray) -> FloatArray:
        def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
            p_tot = max(1e-9, float(y[0]))
            _, _, _, _, _, ple = solve_ternary_equilibrium_point(
                l0, p_tot, spec.e3_total_uM, spec.kd_poi_uM, spec.kd_e3_uM, alpha
            )
            dp_dt = spec.k_syn_uM_per_h - spec.k_deg_basal_per_h * p_tot - spec.k_ub_per_h * ple
            return np.array([dp_dt], dtype=np.float64)

        sol = solve_ivp(
            _rhs,
            (0.0, float(t_eval[-1])),
            np.array([p0], dtype=np.float64),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        clipped: FloatArray = np.asarray(np.clip(sol.y[0] / max(1e-9, p0), 0.0, 1.2), dtype=np.float64)
        return clipped

    times_h = np.linspace(0.0, hours, 97)
    poi_frac = _simulate_at_dose(chosen_dose, times_h)

    # Dose-response sweep at t = hours to compute Dmax and DC50
    dose_grid = np.logspace(-4, 2.0, 25)
    t_end_grid = np.array([0.0, hours], dtype=np.float64)
    degraded_frac = np.zeros(len(dose_grid), dtype=np.float64)
    for i, d_uM in enumerate(dose_grid):
        rem = _simulate_at_dose(float(d_uM), t_end_grid)[-1]
        degraded_frac[i] = max(0.0, 1.0 - float(rem))

    dmax = float(np.max(degraded_frac))
    # DC50: lowest concentration reaching 50% of Dmax (or 50% total degradation if Dmax >= 0.5)
    target_deg = 0.5 if dmax >= 0.5 else 0.5 * dmax
    dc50 = float(dose_grid[-1])
    peak_i = int(np.argmax(degraded_frac))
    for i in range(peak_i + 1):
        if degraded_frac[i] >= target_deg:
            if i == 0:
                dc50 = float(dose_grid[0])
            else:
                x0, x1 = math.log10(dose_grid[i - 1]), math.log10(dose_grid[i])
                y0, y1 = degraded_frac[i - 1], degraded_frac[i]
                frac = (target_deg - y0) / max(1e-9, y1 - y0)
                dc50 = 10.0 ** (x0 + frac * (x1 - x0))
            break

    aoc = float(np.sum(0.5 * ((1.0 - poi_frac[:-1]) + (1.0 - poi_frac[1:])) * np.diff(times_h)) / max(1e-9, hours))
    return ProtacDegradationResult(
        curve=curve,
        dose_uM=float(chosen_dose),
        times_h=times_h,
        poi_fraction_remaining=poi_frac,
        dose_grid_uM=dose_grid,
        dmax_fraction=dmax,
        dc50_uM=dc50,
        degradation_aoc=aoc,
    )
