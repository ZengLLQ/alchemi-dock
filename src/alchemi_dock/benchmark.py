"""Curated medicinal chemistry benchmark libraries and ground-truth SAR oracles.

Provides structurally diverse kinase, GPCR/CNS, matched molecular pair, and PROTAC
degrader libraries with deterministic biophysical ground-truth SAR labels for
validating GNN scaffold generalization, MMP mining, docking enrichment, and MCTS optimization.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .descriptors import (
    compute_clogp_and_mr,
    compute_tpsa,
    count_hbd_hba,
)
from .molgraph import match_smarts, parse_smiles
from .types import FloatArray, ProtacSpec

# 36 curated drug-like and kinase-focused SMILES spanning multiple Murcko scaffolds
CURATED_LIGAND_LIBRARY: Tuple[Tuple[str, str], ...] = (
    ("Erlotinib_Core", "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC"),
    ("Gefitinib_Core", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),
    ("Lapatinib_Fragment", "Fc1ccc(COc2ccc(Nc3ncnc4ccc(-c5ccco5)cc34)cc2Cl)cc1"),
    ("Imatinib_Analog", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("Nilotinib_Analog", "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1"),
    ("Dasatinib_Analog", "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1"),
    ("Quinazoline_H", "c1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_F", "Fc1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_Cl", "Clc1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_OMe", "COc1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_CF3", "FC(F)(F)c1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_OH", "Oc1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_CN", "N#Cc1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_Me", "Cc1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Pyrimidine_H", "c1ccc(Nc2ncccn2)cc1"),
    ("Pyrimidine_F", "Fc1ccc(Nc2ncccn2)cc1"),
    ("Pyrimidine_Cl", "Clc1ccc(Nc2ncccn2)cc1"),
    ("Pyrimidine_OMe", "COc1ccc(Nc2ncccn2)cc1"),
    ("Pyrimidine_CN", "N#Cc1ccc(Nc2ncccn2)cc1"),
    ("Pyrimidine_Morph", "c1ccc(Nc2nc(N3CCOCC3)ccn2)cc1"),
    ("Indazole_Scaffold", "Cc1ccc2c(c1)c(N)nn2"),
    ("Indazole_Benzamide", "O=C(Nc1ccc2[nH]ncc2c1)c1ccccc1"),
    ("Indazole_Fluorobenz", "O=C(Nc1ccc2[nH]ncc2c1)c1ccc(F)cc1"),
    ("Indazole_Pyridine", "O=C(Nc1ccc2[nH]ncc2c1)c1ccncc1"),
    ("Benzimidazole_Core", "c1ccc2[nH]c(-c3ccncc3)nc2c1"),
    ("Benzimidazole_F", "Fc1ccc2[nH]c(-c3ccncc3)nc2c1"),
    ("Benzimidazole_Cl", "Clc1ccc2[nH]c(-c3ccncc3)nc2c1"),
    ("Benzimidazole_Amide", "NC(=O)c1ccc2[nH]c(-c3ccncc3)nc2c1"),
    ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("Diazepam_Scaffold", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
    ("Sulfa_Analog", "Nc1ccc(S(=O)(=O)Nc2ccccn2)cc1"),
    ("Pyrrolopyrimidine_H", "c1ccc(Nc2ncnc3[nH]ccc23)cc1"),
    ("Pyrrolopyrimidine_F", "Fc1ccc(Nc2ncnc3[nH]ccc23)cc1"),
)


def compute_planted_sar_pic50(smiles: str, noise: float = 0.0) -> float:
    """Deterministic biophysical ground-truth SAR oracle for kinase hinge-binder benchmarks.

    Encodes interpretable structure-activity relationships:
    - Hinge-binding bicyclic heteroaromatic motif (quinazoline/pyrimidine/indazole)
    - Optimal lipophilicity cLogP around 3.2
    - Hydrogen-bond donor/acceptor balance
    - Para-halogen / nitrile boost vs carboxylic acid penalty inside hydrophobic ATP pocket
    """
    mol = parse_smiles(smiles)
    clogp, _ = compute_clogp_and_mr(mol)
    tpsa = compute_tpsa(mol)
    hbd, hba = count_hbd_hba(mol)

    has_hinge_ncn = 1.15 if match_smarts(mol, "c:n:c:n") else (0.75 if match_smarts(mol, "c:n:c") else 0.0)
    has_nh_hinge = 0.55 if match_smarts(mol, "[NH]c1ncnc2ccccc12") or match_smarts(mol, "c[NH]c") else 0.0
    has_halogen = 0.38 if any(a.element in ("F", "Cl") for a in mol.atoms) else 0.0
    has_nitrile = 0.42 if match_smarts(mol, "C#N") else 0.0
    has_acid = -1.25 if match_smarts(mol, "C(=O)[OH]") else 0.0

    lip_term = 1.10 * np.exp(-((clogp - 3.2) / 1.8) ** 2)
    hbond_term = 0.16 * min(4, hbd) + 0.12 * min(6, hba)
    tpsa_pen = -0.012 * max(0.0, tpsa - 105.0)
    ring_bonus = 0.25 * min(4, len(mol.sssr_rings))

    score = (
        4.65
        + has_hinge_ncn
        + has_nh_hinge
        + has_halogen
        + has_nitrile
        + has_acid
        + lip_term
        + hbond_term
        + tpsa_pen
        + ring_bonus
    )
    return float(score + noise)


def load_kinase_benchmark_dataset(seed: int = 42) -> Tuple[List[str], List[str], FloatArray]:
    """Return (names, smiles_list, target_pic50) for the 36-ligand benchmark set."""
    rng = np.random.default_rng(seed)
    names = [item[0] for item in CURATED_LIGAND_LIBRARY]
    smiles = [item[1] for item in CURATED_LIGAND_LIBRARY]
    targets = np.array(
        [compute_planted_sar_pic50(smi, noise=float(rng.normal(0.0, 0.05))) for smi in smiles],
        dtype=np.float64,
    )
    return names, smiles, targets


def default_protac_benchmark_specs() -> Dict[str, ProtacSpec]:
    """Return a curated panel of VHL/CRBN PROTAC linker designs for benchmarking."""
    return {
        "short_clash_linker": ProtacSpec(
            name="PROTAC_Short_Clash_C3",
            warhead_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1",
            linker_smiles="CCCN",
            e3_ligand_smiles="O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
            kd_poi_uM=0.045,
            kd_e3_uM=0.120,
        ),
        "optimal_rigid_linker": ProtacSpec(
            name="PROTAC_Rigid_Piperazine_C10",
            warhead_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1",
            linker_smiles="CCN1CCN(CC(=O)NC)CC1",
            e3_ligand_smiles="O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
            kd_poi_uM=0.045,
            kd_e3_uM=0.120,
        ),
        "long_peg_linker": ProtacSpec(
            name="PROTAC_Long_PEG4",
            warhead_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1",
            linker_smiles="COCCOCCOCCOCCOCC(=O)NC",
            e3_ligand_smiles="O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
            kd_poi_uM=0.045,
            kd_e3_uM=0.120,
        ),
    }
