"""Tests for pure-Python SMILES/SMARTS parsing, SSSR, Huckel aromaticity, Kekulization,
Gasteiger partial charges, and 3D conformer generation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alchemi_dock.molgraph import (
    SmilesParseError,
    canonical_smiles,
    compute_gasteiger_charges,
    compute_sssr,
    formal_charge_sum,
    generate_conformer_3d,
    match_smarts,
    molecular_formula,
    molecular_weight,
    parse_smiles,
)


def test_smiles_parsing_and_formula_conservation() -> None:
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
    assert aspirin.num_atoms == 13
    assert aspirin.num_bonds == 13
    assert molecular_formula(aspirin) == "C9H8O4"
    assert abs(molecular_weight(aspirin) - 180.159) < 0.1
    assert formal_charge_sum(aspirin) == 0


def test_smiles_bracket_atoms_and_disconnected_fragments() -> None:
    salt = parse_smiles("[NH3+]CC(=O)[O-].[Na+]" if False else "[NH3+]CC(=O)[O-].Cl")
    assert len(salt.atoms) == 6
    assert formal_charge_sum(salt) == 0
    assert salt.atoms[0].formal_charge == 1
    assert salt.atoms[4].formal_charge == -1


def test_invalid_smiles_raises_clean_error() -> None:
    for bad in ("", "C(C", "C1CC", "[NH3+"):
        with pytest.raises(SmilesParseError):
            parse_smiles(bad)


def test_canonical_smiles_permutation_invariance() -> None:
    variants = [
        "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC",
        "c1(OC)c(OC)cc2ncnc(Nc3cc(C#C)ccc3)c2c1",
        "C#Cc1cccc(Nc2ncnc3cc(OC)c(OC)cc23)c1",
    ]
    canons = {canonical_smiles(parse_smiles(v)) for v in variants}
    assert len(canons) == 1


def test_sssr_gf2_euler_cycle_rank_invariant() -> None:
    cases = [
        ("c1ccccc1", 1),
        ("c1ccc2ccccc2c1", 2),
        ("C12C3C4C1C5C2C3C45", 5),  # Cubane: E=12, V=8 -> E - V + 1 = 5
        ("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", 4),
    ]
    for smi, expected_rings in cases:
        mol = parse_smiles(smi)
        rings = compute_sssr(mol)
        assert len(rings) == expected_rings


def test_huckel_aromaticity_from_kekule_and_kekulization() -> None:
    kek_benzene = parse_smiles("C1=CC=CC=C1")
    assert all(a.is_aromatic for a in kek_benzene.atoms)
    double_bonds = sum(1 for b in kek_benzene.bonds if abs(b.order - 2.0) < 1e-6)
    single_bonds = sum(1 for b in kek_benzene.bonds if abs(b.order - 1.0) < 1e-6)
    assert double_bonds == 3
    assert single_bonds == 3

    cyclobutadiene = parse_smiles("C1=CC=C1")
    assert not any(a.is_aromatic for a in cyclobutadiene.atoms)

    indole = parse_smiles("c1ccc2[nH]ccc2c1")
    assert all(a.is_aromatic for a in indole.atoms)
    assert sum(1 for b in indole.bonds if abs(b.order - 2.0) < 1e-6) == 4


def test_smarts_substructure_matching() -> None:
    quinazoline = parse_smiles("Fc1ccc(Nc2ncnc3ccccc23)cc1")
    assert len(match_smarts(quinazoline, "c:n:c:n")) >= 1
    assert len(match_smarts(quinazoline, "cF")) == 1
    assert len(match_smarts(quinazoline, "[N+](=O)[O-]")) == 0

    nitro_mol = parse_smiles("c1ccc([N+](=O)[O-])cc1")
    assert len(match_smarts(nitro_mol, "[N+](=O)[O-]")) == 1

    epoxide = parse_smiles("C1OC1C")
    assert len(match_smarts(epoxide, "[C;r3]1[O,N;r3][C;r3]1")) == 1


def test_gasteiger_charge_conservation_and_polarity() -> None:
    for smi in ("CC(=O)O", "Fc1ccc(N)cc1", "[NH3+]CC(=O)[O-]", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"):
        mol = parse_smiles(smi)
        q = compute_gasteiger_charges(mol)
        assert abs(float(np.sum(q)) - formal_charge_sum(mol)) < 1e-9

    fluoromethane = parse_smiles("CF")
    # Fluorine must be more negative than Carbon
    assert fluoromethane.atoms[1].partial_charge < fluoromethane.atoms[0].partial_charge


def test_3d_conformer_bond_lengths_realistic() -> None:
    caffeine = parse_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    conf = generate_conformer_3d(caffeine, seed=42, add_hydrogens=True)
    assert conf.coords.shape[1] == 3
    assert conf.strain_energy < 25.0
    for u, v, _ in conf.bonds:
        d = float(np.linalg.norm(conf.coords[u] - conf.coords[v]))
        assert 0.85 <= d <= 1.85
    assert math.isfinite(conf.strain_energy)
