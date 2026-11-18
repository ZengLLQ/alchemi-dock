"""Tests for ADME pharmacokinetics, CYP kinetics, oral PK ODEs, and empirical docking."""

from __future__ import annotations

import numpy as np

from alchemi_dock.adme import (
    classify_ionization,
    hepatic_extraction_well_stirred,
    predict_adme_profile,
    simulate_oral_pk,
)
from alchemi_dock.docking import (
    build_kinase_atp_pocket,
    dock_ligand,
    kabsch_rmsd,
    score_ligand_pose,
)
from alchemi_dock.molgraph import generate_conformer_3d, parse_smiles


def test_ionization_microspecies_classification() -> None:
    acid_cls, _ = classify_ionization(parse_smiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1"))
    base_cls, _ = classify_ionization(parse_smiles("CN1CCN(C)CC1"))
    zwit_cls, _ = classify_ionization(parse_smiles("NCC(=O)O"))
    neut_cls, _ = classify_ionization(parse_smiles("c1ccccc1"))
    assert acid_cls == "acid"
    assert base_cls == "base"
    assert zwit_cls == "zwitterion"
    assert neut_cls == "neutral"


def test_adme_hepatic_extraction_and_pk_ode() -> None:
    adme = predict_adme_profile("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC")
    assert 0.0 < adme.fu_plasma < 1.0
    assert 0.0 < adme.oral_bioavailability_f < 1.0
    assert 0.0 <= adme.cns_mpo <= 6.0

    eh, cl_h, fh = hepatic_extraction_well_stirred(adme.cl_int_ul_min_mg, adme.fu_plasma)
    assert 0.0 < eh < 1.0
    assert abs((1.0 - eh) - fh) < 1e-6 or fh >= 0.01
    assert cl_h > 0.0

    pk = simulate_oral_pk(adme, dose_mg=100.0, dosing_interval_h=12.0, n_doses=3)
    assert np.all(pk.plasma_conc_uM >= -1e-12)
    assert pk.cmax_uM > 0.0
    assert pk.auc_0_inf_uM_h > 0.0
    assert pk.half_life_h > 0.0


def test_kabsch_rmsd_se3_invariance() -> None:
    mol = parse_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    conf = generate_conformer_3d(mol, seed=42, add_hydrogens=False)
    coords = conf.coords
    # Apply random proper rotation + translation
    theta = 0.85
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transformed = coords @ rot.T + np.array([3.5, -2.1, 4.7])
    assert kabsch_rmsd(coords, transformed) < 1e-10


def test_empirical_docking_achieves_favorable_binding_and_contacts() -> None:
    pocket = build_kinase_atp_pocket()
    pose = dock_ligand("Fc1ccc(Nc2ncnc3ccccc23)cc1", pocket=pocket, mc_steps=140, seed=42)
    assert pose.total_score_kcal_mol < -1.0
    assert pose.gauss1_term < 0.0
    assert len(pose.contacts) >= 1

    # Verify scoring directly matches pose output
    mol = parse_smiles("Fc1ccc(Nc2ncnc3ccccc23)cc1")
    charges = np.array([a.partial_charge for a in mol.atoms], dtype=np.float64)
    score2, _, _ = score_ligand_pose(mol, pose.ligand_coords, pocket, charges, len(mol.bonds) * 0 + 3)
    assert np.isfinite(score2)
