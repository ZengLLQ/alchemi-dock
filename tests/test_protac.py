"""Tests for PROTAC ternary complex equilibrium, Hook effect, and degradation ODEs."""

from __future__ import annotations

import numpy as np

from alchemi_dock.benchmark import default_protac_benchmark_specs
from alchemi_dock.protac import (
    compute_protac_equilibrium_curve,
    estimate_protac_cooperativity,
    simulate_protac_degradation,
    solve_ternary_equilibrium_point,
)
from alchemi_dock.types import ProtacSpec


def test_douglass_miller_exact_mass_conservation() -> None:
    p0, e0, kd_p, kd_e, alpha = 0.50, 0.20, 0.05, 0.12, 8.0
    for l0 in (1e-4, 1e-2, 0.1, 0.5, 5.0, 100.0):
        lf, pf, ef, pl, el, ple = solve_ternary_equilibrium_point(l0, p0, e0, kd_p, kd_e, alpha)
        assert min(lf, pf, ef, pl, el, ple) >= 0.0
        assert abs((pf + pl + ple) - p0) < 1e-11
        assert abs((ef + el + ple) - e0) < 1e-11
        assert abs((lf + pl + el + ple) - l0) < 1e-11


def test_protac_hook_effect_and_cooperativity_amplification() -> None:
    specs = default_protac_benchmark_specs()
    short_alpha = estimate_protac_cooperativity(specs["short_clash_linker"].linker_smiles)
    opt_alpha = estimate_protac_cooperativity(specs["optimal_rigid_linker"].linker_smiles)
    assert opt_alpha > short_alpha

    low_coop = ProtacSpec(
        name="LowAlpha",
        warhead_smiles=specs["optimal_rigid_linker"].warhead_smiles,
        linker_smiles="CCCN",
        e3_ligand_smiles=specs["optimal_rigid_linker"].e3_ligand_smiles,
        kd_poi_uM=0.05,
        kd_e3_uM=0.10,
        cooperativity_alpha=0.5,
    )
    high_coop = ProtacSpec(
        name="HighAlpha",
        warhead_smiles=specs["optimal_rigid_linker"].warhead_smiles,
        linker_smiles="CCN1CCN(CC(=O)NC)CC1",
        e3_ligand_smiles=specs["optimal_rigid_linker"].e3_ligand_smiles,
        kd_poi_uM=0.05,
        kd_e3_uM=0.10,
        cooperativity_alpha=12.0,
    )
    c_low = compute_protac_equilibrium_curve(low_coop)
    c_high = compute_protac_equilibrium_curve(high_coop)
    assert c_high.peak_ternary_uM > c_low.peak_ternary_uM
    assert c_high.mass_balance_max_err < 1e-11

    # Verify bell-shaped Hook effect: peak exceeds both extreme left and extreme right
    assert c_high.peak_ternary_uM > float(c_high.ternary_ple_uM[0]) * 5.0
    assert c_high.peak_ternary_uM > float(c_high.ternary_ple_uM[-1]) * 5.0


def test_protac_degradation_kinetics_and_zero_dose_baseline() -> None:
    spec = default_protac_benchmark_specs()["optimal_rigid_linker"]
    deg_active = simulate_protac_degradation(spec, hours=24.0)
    assert 0.25 < deg_active.dmax_fraction <= 1.0
    assert 1e-4 <= deg_active.dc50_uM <= 10.0
    assert deg_active.degradation_aoc > 0.0

    deg_zero = simulate_protac_degradation(spec, dose_uM=0.0, hours=24.0)
    assert np.allclose(deg_zero.poi_fraction_remaining, 1.0, atol=1e-4)
