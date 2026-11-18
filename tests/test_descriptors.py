"""Tests for physicochemical descriptors, fingerprints, Butina clustering, scaffolds, and MMPs."""

from __future__ import annotations

import numpy as np

from alchemi_dock.descriptors import (
    bemis_murcko_scaffold,
    butina_clustering,
    compute_clogp_and_mr,
    compute_fingerprint_set,
    compute_morgan_fingerprint,
    compute_qed,
    compute_sa_score,
    compute_topological_indices,
    compute_tpsa,
    count_hbd_hba,
    count_lipinski_violations,
    detect_structural_alerts,
    maxmin_diversity_pick,
    mine_matched_molecular_pairs,
    similarity_matrix,
    tanimoto_similarity,
)
from alchemi_dock.molgraph import parse_smiles


def test_clogp_and_tpsa_physical_trends() -> None:
    hexane = parse_smiles("CCCCCC")
    ethanol = parse_smiles("CCO")
    acetamide = parse_smiles("CC(=O)N")

    lp_hex, _ = compute_clogp_and_mr(hexane)
    lp_eth, _ = compute_clogp_and_mr(ethanol)
    lp_ace, _ = compute_clogp_and_mr(acetamide)
    assert lp_hex > lp_eth > lp_ace

    assert compute_tpsa(hexane) == 0.0
    assert abs(compute_tpsa(ethanol) - 20.23) < 0.05
    assert abs(compute_tpsa(acetamide) - 43.09) < 0.05


def test_qed_ro5_sa_and_alerts() -> None:
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    qed_asp = compute_qed(aspirin)
    assert 0.4 <= qed_asp <= 0.9
    hbd, hba = count_hbd_hba(aspirin)
    assert hbd == 1
    assert hba == 4
    assert count_lipinski_violations(180.16, 1.3, hbd, hba) == 0
    assert detect_structural_alerts(aspirin) == []

    michael = parse_smiles("C=CC(=O)Nc1ccccc1")
    assert "michael_acceptor" in detect_structural_alerts(michael)

    macro = parse_smiles("C1CCCCCCCCCC1C2(CC2)C3(CCC3)C(=O)O")
    assert compute_sa_score(macro) > compute_sa_score(aspirin)


def test_topological_indices_and_morgan_similarity() -> None:
    butane = parse_smiles("CCCC")
    idx_but = compute_topological_indices(butane)
    assert idx_but["wiener"] == 10.0  # 1+2+3 + 1+2 + 1 = 10

    m1 = parse_smiles("Fc1ccc(Nc2ncnc3ccccc23)cc1")
    m2 = parse_smiles("Clc1ccc(Nc2ncnc3ccccc23)cc1")
    m3 = parse_smiles("CCCCCC(=O)O")

    fp1 = compute_fingerprint_set(m1)
    fp2 = compute_fingerprint_set(m2)
    fp3 = compute_fingerprint_set(m3)

    assert tanimoto_similarity(fp1.ecfp4_bits, fp1.ecfp4_bits) == 1.0
    assert tanimoto_similarity(fp1.ecfp4_bits, fp2.ecfp4_bits) > tanimoto_similarity(fp1.ecfp4_bits, fp3.ecfp4_bits)

    bits_r2, counts_r2 = compute_morgan_fingerprint(m1, radius=2, n_bits=1024)
    assert int(np.sum(bits_r2)) > 10
    assert sum(counts_r2.values()) >= int(np.sum(bits_r2))


def test_butina_clustering_and_maxmin_diversity() -> None:
    smis = [
        "Fc1ccc(Nc2ncnc3ccccc23)cc1",
        "Clc1ccc(Nc2ncnc3ccccc23)cc1",
        "COc1ccc(Nc2ncnc3ccccc23)cc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    ]
    fps = [compute_fingerprint_set(parse_smiles(s)).ecfp4_bits for s in smis]
    sim = similarity_matrix(fps, metric="tanimoto")
    assert sim.shape == (5, 5)
    assert np.allclose(sim, sim.T)

    clusters = butina_clustering(fps, similarity_cutoff=0.50)
    flat = sorted(idx for cl in clusters for idx in cl)
    assert flat == [0, 1, 2, 3, 4]
    assert len(clusters) < 5  # quinazoline analogs must group together

    picked = maxmin_diversity_pick(fps, k=3, seed_idx=0)
    assert len(picked) == 3
    assert len(set(picked)) == 3


def test_bemis_murcko_and_mmp_mining() -> None:
    gefitinib = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
    scaff = bemis_murcko_scaffold(gefitinib)
    assert scaff.ring_system_count == 3
    assert scaff.sidechain_atom_count >= 4
    assert len(scaff.murcko_smiles) < len(gefitinib.canonical_smiles)
    assert len(scaff.generic_skeleton_smiles) > 0

    series = [
        "c1ccc(Nc2ncnc3ccccc23)cc1",
        "Fc1ccc(Nc2ncnc3ccccc23)cc1",
        "Clc1ccc(Nc2ncnc3ccccc23)cc1",
        "N#Cc1ccc(Nc2ncnc3ccccc23)cc1",
    ]
    props = [6.1, 6.6, 6.5, 6.9]
    mmps = mine_matched_molecular_pairs(series, props, property_name="pIC50")
    assert len(mmps) >= 3
    assert any(abs(m.delta_property) > 0.3 for m in mmps)
