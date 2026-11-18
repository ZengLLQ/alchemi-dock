"""Physicochemical descriptors, circular Morgan fingerprints, Butina clustering,
Bemis-Murcko scaffold decomposition, and Matched Molecular Pair (MMP) analysis.

Implements Wildman-Crippen logP/MR, Ertl TPSA, Bickerton QED, Lipinski/Veber rules,
SAscore synthetic accessibility, ECFP4/ECFP6/FCFP4 circular hashing, MACCS structural
keys, Taylor-Butina leader-follower clustering, and Hussain-Rea MMP transformations.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .molgraph import (
    canonical_smiles,
    identify_rotatable_bonds,
    match_smarts,
    molecular_weight,
    parse_smiles,
    perceive_molecule,
)
from .types import (
    ATOMIC_NUMBERS,
    QED_PARAMS,
    STRUCTURAL_ALERTS,
    AtomRecord,
    BondRecord,
    FingerprintSet,
    FloatArray,
    IntArray,
    MMPTransform,
    MoleculeGraph,
    ScaffoldDecomposition,
)


def _crippen_atom_contrib(mol: MoleculeGraph, idx: int) -> Tuple[float, float]:
    """Return (logP_contrib, MR_contrib) for heavy atom idx and its attached H atoms.

    Based on Wildman-Crippen (1999) atomic contribution environments.
    """
    atom = mol.atoms[idx]
    nbrs = mol.neighbor_bonds(idx)
    elem = atom.element
    h_cnt = atom.implicit_h
    arom_nbrs = sum(1 for n_idx, _ in nbrs if mol.atoms[n_idx].is_aromatic)
    hetero_nbrs = sum(1 for n_idx, _ in nbrs if mol.atoms[n_idx].element not in ("C", "H"))
    double_o = any(
        mol.atoms[n_idx].element == "O" and b.order >= 1.8 for n_idx, b in nbrs
    )
    double_n = any(
        mol.atoms[n_idx].element == "N" and b.order >= 1.8 for n_idx, b in nbrs
    )

    logp = 0.0
    mr = 0.0
    if elem == "C":
        if atom.is_aromatic:
            if hetero_nbrs > 0:
                logp, mr = 0.1441, 3.350
            elif len(nbrs) == 3:
                logp, mr = 0.2960, 3.760
            else:
                logp, mr = 0.1581, 3.350
        elif atom.hybridization == "sp":
            logp, mr = 0.0017, 3.880
        elif atom.hybridization == "sp2":
            if double_o or double_n:
                logp, mr = -0.2783, 3.040
            elif hetero_nbrs > 0:
                logp, mr = -0.0249, 3.410
            else:
                logp, mr = 0.1551, 3.500
        else:
            if hetero_nbrs == 0:
                logp = 0.1441 if h_cnt >= 2 else (0.1276 if h_cnt == 1 else 0.0813)
                mr = 2.503
            elif hetero_nbrs == 1:
                logp, mr = -0.2035, 2.753
            else:
                logp, mr = -0.4530, 2.864
        # Attached hydrogen contribution
        logp += h_cnt * 0.1230
        mr += h_cnt * 1.057
    elif elem == "N":
        if atom.formal_charge > 0:
            logp, mr = -1.9500, 2.200
        elif atom.is_aromatic:
            logp, mr = (-0.3239, 2.300) if h_cnt > 0 else (-0.4930, 2.770)
        elif any(b.order >= 2.5 for _, b in nbrs):
            logp, mr = -0.3187, 3.120
        elif arom_nbrs > 0 or double_o:
            logp, mr = -0.7096, 3.000
        else:
            logp = -1.0190 if h_cnt >= 2 else (-0.7096 if h_cnt == 1 else -0.3187)
            mr = 2.870
        logp += h_cnt * -0.2677
        mr += h_cnt * 0.890
    elif elem == "O":
        if atom.formal_charge < 0:
            logp, mr = -1.1800, 1.850
        elif atom.is_aromatic:
            logp, mr = 0.1552, 1.630
        elif any(b.order >= 1.8 for _, b in nbrs):
            logp, mr = -0.3280, 1.361  # carbonyl / sulfinyl / nitro oxygen
        elif h_cnt > 0:
            # Alcohol vs carboxylic/phenol OH
            attached_carbonyl = any(
                any(mol.atoms[nn].element == "O" and bb.order >= 1.8 for nn, bb in mol.neighbor_bonds(n_idx))
                for n_idx, _ in nbrs
            )
            logp = -0.2893 if not attached_carbonyl else -0.4900
            mr = 1.630
        else:
            logp, mr = -0.0680, 1.630
        logp += h_cnt * -0.2893
        mr += h_cnt * 0.890
    elif elem == "S":
        if atom.is_aromatic:
            logp, mr = 0.6237, 7.591
        elif double_o:
            logp, mr = -0.1800, 7.365
        else:
            logp, mr = 0.6482, 7.591
        logp += h_cnt * 0.1230
        mr += h_cnt * 1.057
    elif elem == "P":
        logp, mr = 0.8612, 6.920
    elif elem == "F":
        logp, mr = 0.4118, 0.920
    elif elem == "Cl":
        logp, mr = 0.6895, 5.853
    elif elem == "Br":
        logp, mr = 0.8456, 8.927
    elif elem == "I":
        logp, mr = 1.1500, 14.020
    elif elem == "B":
        logp, mr = -0.3800, 4.100
    else:
        logp, mr = 0.0, 2.500

    return logp, mr


def compute_clogp_and_mr(mol: MoleculeGraph) -> Tuple[float, float]:
    """Compute Wildman-Crippen octanol-water partition coefficient (cLogP) and Molar Refractivity."""
    total_logp = 0.0
    total_mr = 0.0
    for idx in range(mol.num_atoms):
        lp, mr = _crippen_atom_contrib(mol, idx)
        total_logp += lp
        total_mr += mr
    return float(total_logp), float(total_mr)


def compute_tpsa(mol: MoleculeGraph, include_sp: bool = True) -> float:
    """Compute Ertl Topological Polar Surface Area (TPSA, Angstrom^2)."""
    tpsa = 0.0
    for atom in mol.atoms:
        nbr_bonds = mol.neighbor_bonds(atom.index)
        deg = len(nbr_bonds)
        h = atom.implicit_h
        has_triple = any(b.order >= 2.8 for _, b in nbr_bonds)
        n_double = sum(1 for _, b in nbr_bonds if 1.8 <= b.order < 2.8)
        in_3ring = 3 in atom.ring_sizes

        if atom.element == "N":
            if atom.formal_charge == 0:
                if atom.is_aromatic:
                    tpsa += 15.79 if h >= 1 else 12.89
                elif has_triple:
                    tpsa += 23.79
                elif n_double >= 1:
                    tpsa += 23.85 if h >= 1 else 12.36
                elif in_3ring:
                    tpsa += 21.94 if h >= 1 else 3.01
                elif deg == 1 and h == 2:
                    tpsa += 26.02
                elif deg == 2 and h == 1:
                    tpsa += 12.03
                else:
                    tpsa += 3.24
            elif atom.formal_charge > 0:
                if atom.is_aromatic:
                    tpsa += 3.88
                elif h == 3:
                    tpsa += 27.64
                elif h == 2:
                    tpsa += 16.61
                elif h == 1:
                    tpsa += 4.44
                else:
                    tpsa += 0.0
        elif atom.element == "O":
            if atom.formal_charge < 0:
                tpsa += 23.06
            elif atom.is_aromatic:
                tpsa += 13.14
            elif n_double >= 1:
                tpsa += 17.07
            elif in_3ring:
                tpsa += 12.53
            elif h >= 1:
                tpsa += 20.23
            else:
                tpsa += 9.23
        elif include_sp and atom.element == "S":
            if atom.is_aromatic:
                tpsa += 28.24
            elif n_double >= 1:
                tpsa += 32.09 if h == 0 and deg == 1 else 17.07
            elif h >= 1:
                tpsa += 38.80
            else:
                tpsa += 25.30
        elif include_sp and atom.element == "P":
            if n_double >= 1:
                tpsa += 9.81
            else:
                tpsa += 13.59
    return float(tpsa)


def count_hbd_hba(mol: MoleculeGraph) -> Tuple[int, int]:
    """Count Lipinski hydrogen bond donors (OH/NH) and acceptors (N/O)."""
    hbd = 0
    hba = 0
    for atom in mol.atoms:
        if atom.element in ("N", "O"):
            hba += 1
            if atom.implicit_h > 0:
                hbd += atom.implicit_h
    return hbd, hba


def compute_fsp3(mol: MoleculeGraph) -> float:
    """Compute fraction of sp3-hybridized carbons (Fsp3)."""
    carbons = [a for a in mol.atoms if a.element == "C"]
    if not carbons:
        return 0.0
    sp3_c = sum(1 for a in carbons if a.hybridization == "sp3" and not a.is_aromatic)
    return float(sp3_c / len(carbons))


def compute_topological_indices(mol: MoleculeGraph) -> Dict[str, float]:
    """Compute Wiener index, Balaban J index, and Bertz-like complexity index."""
    n = mol.num_atoms
    if n <= 1:
        return {"wiener": 0.0, "balaban_j": 0.0, "bertz_complexity": 0.0}

    # Shortest-path distance matrix via BFS
    dist = np.full((n, n), fill_value=0.0, dtype=np.float64)
    adj: List[List[int]] = [[] for _ in range(n)]
    for b in mol.bonds:
        adj[b.atom_u].append(b.atom_v)
        adj[b.atom_v].append(b.atom_u)

    for src in range(n):
        seen = {src: 0}
        q: deque[int] = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    dist[src, v] = float(seen[v])
                    q.append(v)

    wiener = float(0.5 * np.sum(dist))
    s_dist = np.sum(dist, axis=1)
    q_edges = mol.num_bonds
    mu = max(0, q_edges - n + 1)
    j_sum = 0.0
    for b in mol.bonds:
        denom = math.sqrt(max(1e-9, s_dist[b.atom_u] * s_dist[b.atom_v]))
        j_sum += 1.0 / denom
    balaban_j = float((q_edges / (mu + 1.0)) * j_sum) if q_edges > 0 else 0.0

    # Heteroatom + bond-order + branching entropy proxy for Bertz complexity
    elem_counts: Dict[str, int] = {}
    for a in mol.atoms:
        key = f"{a.element}_{a.hybridization}_{int(a.is_aromatic)}"
        elem_counts[key] = elem_counts.get(key, 0) + 1
    entropy = 0.0
    for cnt in elem_counts.values():
        p = cnt / n
        entropy -= p * math.log2(p)
    bond_complexity = sum(b.order * (1.5 if b.in_ring else 1.0) for b in mol.bonds)
    bertz = float(2.0 * bond_complexity * math.log2(max(2.0, float(n))) + n * entropy * 3.5)

    return {
        "wiener": wiener,
        "balaban_j": balaban_j,
        "bertz_complexity": bertz,
    }


def detect_structural_alerts(mol: MoleculeGraph) -> List[str]:
    """Screen a MoleculeGraph against canonical PAINS/Brenk SMARTS alerts."""
    triggered: List[str] = []
    for name, smarts, _ in STRUCTURAL_ALERTS:
        if match_smarts(mol, smarts, uniquify=True):
            triggered.append(name)
    return triggered


def _ads_desirability(x: float, params: Tuple[float, float, float, float, float, float, float, float]) -> float:
    """Bickerton Asymmetric Double Sigmoidal desirability ADS(x) in [0, 1]."""
    _, a, b, c, d, e, f, d_max = params
    exp_e = math.exp(-max(-50.0, min(50.0, (x - c + d / 2.0) / max(1e-9, e))))
    exp_f = math.exp(-max(-50.0, min(50.0, (x - c - d / 2.0) / max(1e-9, f))))
    val = a + (b / (1.0 + exp_e)) * (1.0 - 1.0 / (1.0 + exp_f))
    return float(max(1e-6, min(1.0, val / max(1e-9, d_max))))


def compute_qed(mol: MoleculeGraph, alerts_count: Optional[int] = None) -> float:
    """Compute Bickerton et al. (2012) Quantitative Estimate of Drug-likeness (QED) in [0, 1]."""
    mw = molecular_weight(mol)
    clogp, _ = compute_clogp_and_mr(mol)
    hbd, hba = count_hbd_hba(mol)
    psa = compute_tpsa(mol)
    rotb = len(identify_rotatable_bonds(mol))
    arom_rings = sum(
        1 for cyc in mol.sssr_rings if all(mol.atoms[idx].is_aromatic for idx in cyc)
    )
    alerts = alerts_count if alerts_count is not None else len(detect_structural_alerts(mol))

    props = {
        "MW": mw,
        "ALOGP": clogp,
        "HBA": float(hba),
        "HBD": float(hbd),
        "PSA": psa,
        "ROTB": float(rotb),
        "AROM": float(arom_rings),
        "ALERTS": float(alerts),
    }

    weighted_log_sum = 0.0
    weight_sum = 0.0
    for key, value in props.items():
        params = QED_PARAMS[key]
        w = params[0]
        d_i = _ads_desirability(value, params)
        weighted_log_sum += w * math.log(d_i)
        weight_sum += w
    return float(math.exp(weighted_log_sum / max(1e-9, weight_sum)))


def compute_sa_score(mol: MoleculeGraph) -> float:
    """Compute Ertl-style Synthetic Accessibility score in [1.0 (easy), 10.0 (hard)]."""
    n = max(1, mol.num_atoms)
    n_rings = len(mol.sssr_rings)
    spiro_or_bridge = 0
    ring_membership: Dict[int, int] = {}
    for cyc in mol.sssr_rings:
        for a in cyc:
            ring_membership[a] = ring_membership.get(a, 0) + 1
    for cnt in ring_membership.values():
        if cnt >= 2:
            spiro_or_bridge += cnt - 1

    stereo_centers = sum(
        1
        for a in mol.atoms
        if a.element == "C"
        and a.hybridization == "sp3"
        and len(mol.neighbors(a.index)) >= 3
        and a.implicit_h <= 1
    )
    macrocycles = sum(1 for cyc in mol.sssr_rings if len(cyc) >= 8)
    hetero_frac = sum(1 for a in mol.atoms if a.element not in ("C", "H")) / n
    size_penalty = max(0.0, (n - 18) * 0.045)
    ring_penalty = 0.35 * spiro_or_bridge + 0.9 * macrocycles + 0.12 * max(0, n_rings - 3)
    stereo_penalty = 0.22 * stereo_centers
    fragment_bonus = max(0.0, 1.4 - 0.8 * abs(hetero_frac - 0.22))
    raw = 1.6 + size_penalty + ring_penalty + stereo_penalty - 0.3 * fragment_bonus
    return float(max(1.0, min(10.0, raw)))


def count_lipinski_violations(mw: float, clogp: float, hbd: int, hba: int) -> int:
    """Count Lipinski Rule of 5 violations (MW>500, cLogP>5, HBD>5, HBA>10)."""
    violations = 0
    if mw > 500.0:
        violations += 1
    if clogp > 5.0:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    return violations


# ---------------------------------------------------------------------------
# Morgan Circular ECFP Fingerprints & MACCS Structural Keys
# ---------------------------------------------------------------------------

def _stable_hash32(data: Tuple[int, ...]) -> int:
    raw = ",".join(str(x) for x in data).encode("ascii")
    digest = hashlib.blake2b(raw, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _atom_invariant(mol: MoleculeGraph, idx: int, pharmacophore: bool = False) -> int:
    atom = mol.atoms[idx]
    if pharmacophore:
        is_hbd = 1 if (atom.element in ("N", "O") and atom.implicit_h > 0) else 0
        is_hba = 1 if atom.element in ("N", "O") else 0
        is_pos = 1 if atom.formal_charge > 0 else 0
        is_neg = 1 if atom.formal_charge < 0 else 0
        is_arom = 1 if atom.is_aromatic else 0
        is_hal = 1 if atom.element in ("F", "Cl", "Br", "I") else 0
        return _stable_hash32((is_hbd, is_hba, is_pos, is_neg, is_arom, is_hal))

    deg = len(mol.neighbors(idx))
    z = ATOMIC_NUMBERS.get(atom.element, 6)
    return _stable_hash32(
        (
            deg,
            atom.implicit_h,
            z,
            atom.formal_charge + 10,
            1 if atom.in_ring else 0,
            1 if atom.is_aromatic else 0,
        )
    )


def compute_morgan_fingerprint(
    mol: MoleculeGraph,
    radius: int = 2,
    n_bits: int = 1024,
    pharmacophore: bool = False,
) -> Tuple[IntArray, Dict[int, int]]:
    """Compute Morgan circular ECFP/FCFP fingerprint bits and collision-folded counts."""
    n = mol.num_atoms
    bits = np.zeros(n_bits, dtype=np.int64)
    counts: Dict[int, int] = {}
    if n == 0:
        return bits, counts

    identifiers = [_atom_invariant(mol, i, pharmacophore=pharmacophore) for i in range(n)]
    seen_env: Set[Tuple[int, frozenset[Tuple[int, int]]]] = set()

    def _record(ident: int, atom_idx: int, bonds_in_env: Set[Tuple[int, int]]) -> None:
        key = (ident, frozenset(bonds_in_env))
        if key in seen_env:
            return
        seen_env.add(key)
        bit_idx = ident % n_bits
        bits[bit_idx] = 1
        counts[bit_idx] = counts.get(bit_idx, 0) + 1

    atom_bond_sets: List[Set[Tuple[int, int]]] = [set() for _ in range(n)]
    for i in range(n):
        _record(identifiers[i], i, atom_bond_sets[i])

    for r in range(1, radius + 1):
        new_ids: List[int] = []
        new_bond_sets: List[Set[Tuple[int, int]]] = []
        for i in range(n):
            nbr_tuples: List[Tuple[int, int]] = []
            bset = set(atom_bond_sets[i])
            for nbr, bond in mol.neighbor_bonds(i):
                bo_code = 4 if bond.is_aromatic else int(round(bond.order))
                nbr_tuples.append((bo_code, identifiers[nbr]))
                bset.add((min(i, nbr), max(i, nbr)))
                bset |= atom_bond_sets[nbr]
            nbr_tuples.sort()
            flat = [r, identifiers[i]]
            for bo_code, nbr_id in nbr_tuples:
                flat.append(bo_code)
                flat.append(nbr_id)
            hid = _stable_hash32(tuple(flat))
            new_ids.append(hid)
            new_bond_sets.append(bset)
        identifiers = new_ids
        atom_bond_sets = new_bond_sets
        for i in range(n):
            _record(identifiers[i], i, atom_bond_sets[i])

    return bits, counts


_MACCS_SMARTS: Tuple[str, ...] = (
    "c1ccccc1",
    "[#7]",
    "[#8]",
    "[#16]",
    "[F,Cl,Br,I]",
    "C=O",
    "C(=O)O",
    "C(=O)N",
    "C#N",
    "S(=O)(=O)",
    "[OH]",
    "[NH2]",
    "[NH]",
    "c[OH]",
    "c[NH2]",
    "C=C",
    "C#C",
    "c:n",
    "[r3]",
    "[r4]",
    "[r5]",
    "[r6]",
    "[r7]",
    "OCC",
    "NCC",
    "cCl",
    "cF",
    "cOC",
    "[N+](=O)[O-]",
    "P=O",
    "c1ccncc1",
    "c1cncnc1",
)


def compute_maccs_keys(mol: MoleculeGraph, size: int = 166) -> IntArray:
    """Compute a 166-bit structural key fingerprint combining SMARTS pharmacophores and ring/count bits."""
    bits = np.zeros(size, dtype=np.int64)
    for idx, pat in enumerate(_MACCS_SMARTS):
        if idx < size and match_smarts(mol, pat, uniquify=True):
            bits[idx] = 1
    # Structural element/ring-count predicates in upper bits
    offset = len(_MACCS_SMARTS)
    predicates = [
        len(mol.sssr_rings) >= 1,
        len(mol.sssr_rings) >= 2,
        len(mol.sssr_rings) >= 3,
        len(mol.sssr_rings) >= 4,
        sum(1 for a in mol.atoms if a.element == "N") >= 2,
        sum(1 for a in mol.atoms if a.element == "O") >= 2,
        sum(1 for a in mol.atoms if a.element == "S") >= 1,
        sum(1 for a in mol.atoms if a.element in ("F", "Cl", "Br", "I")) >= 2,
        sum(1 for a in mol.atoms if a.is_aromatic) >= 6,
        sum(1 for a in mol.atoms if a.is_aromatic) >= 10,
        len(identify_rotatable_bonds(mol)) >= 4,
        molecular_weight(mol) > 350.0,
    ]
    for j, flag in enumerate(predicates):
        if offset + j < size and flag:
            bits[offset + j] = 1
    return bits


def compute_fingerprint_set(mol: MoleculeGraph, n_bits: int = 1024) -> FingerprintSet:
    """Compute full FingerprintSet (ECFP4, ECFP6, FCFP4, MACCS)."""
    ecfp4_bits, ecfp4_counts = compute_morgan_fingerprint(mol, radius=2, n_bits=n_bits, pharmacophore=False)
    ecfp6_bits, _ = compute_morgan_fingerprint(mol, radius=3, n_bits=n_bits, pharmacophore=False)
    fcfp4_bits, _ = compute_morgan_fingerprint(mol, radius=2, n_bits=n_bits, pharmacophore=True)
    maccs = compute_maccs_keys(mol, size=166)
    return FingerprintSet(
        ecfp4_bits=ecfp4_bits,
        ecfp4_counts=ecfp4_counts,
        ecfp6_bits=ecfp6_bits,
        fcfp4_bits=fcfp4_bits,
        maccs_bits=maccs,
        n_bits=n_bits,
    )


# ---------------------------------------------------------------------------
# Similarity Metrics, Diversity Selection & Taylor-Butina Clustering
# ---------------------------------------------------------------------------

def tanimoto_similarity(fp_a: IntArray, fp_b: IntArray) -> float:
    """Compute exact binary Tanimoto similarity coefficient in [0, 1]."""
    both = int(np.sum((fp_a > 0) & (fp_b > 0)))
    either = int(np.sum((fp_a > 0) | (fp_b > 0)))
    return 1.0 if either == 0 else float(both / either)


def dice_similarity(fp_a: IntArray, fp_b: IntArray) -> float:
    """Compute Sorensen-Dice similarity coefficient in [0, 1]."""
    both = int(np.sum((fp_a > 0) & (fp_b > 0)))
    denom = int(np.sum(fp_a > 0) + np.sum(fp_b > 0))
    return 1.0 if denom == 0 else float(2.0 * both / denom)


def similarity_matrix(fps: Sequence[IntArray], metric: str = "tanimoto") -> FloatArray:
    """Compute vectorized pairwise similarity matrix across a list of binary fingerprints."""
    mat = np.asarray(fps, dtype=np.float64)
    if mat.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    bin_mat = (mat > 0).astype(np.float64)
    inter = bin_mat @ bin_mat.T
    counts = np.sum(bin_mat, axis=1, keepdims=True)
    if metric == "dice":
        denom = counts + counts.T
        sim = np.where(denom > 0, (2.0 * inter) / np.maximum(1.0, denom), 1.0)
    else:
        union = counts + counts.T - inter
        sim = np.where(union > 0, inter / np.maximum(1.0, union), 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def butina_clustering(fps: Sequence[IntArray], similarity_cutoff: float = 0.55) -> List[List[int]]:
    """Taylor-Butina leader-follower clustering over fingerprint Tanimoto similarity."""
    n = len(fps)
    if n == 0:
        return []
    sim = similarity_matrix(fps, metric="tanimoto")
    neighbors: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and sim[i, j] >= similarity_cutoff:
                neighbors[i].add(j)

    # Sort molecules descending by neighbor count (deterministic tie-break by index)
    order = sorted(range(n), key=lambda idx: (-len(neighbors[idx]), idx))
    assigned = [False] * n
    clusters: List[List[int]] = []

    for leader in order:
        if assigned[leader]:
            continue
        cluster = [leader]
        assigned[leader] = True
        for member in sorted(neighbors[leader]):
            if not assigned[member]:
                cluster.append(member)
                assigned[member] = True
        clusters.append(cluster)

    return clusters


def maxmin_diversity_pick(fps: Sequence[IntArray], k: int, seed_idx: int = 0) -> List[int]:
    """Select k maximally diverse molecules using greedy MaxMin Tanimoto distance."""
    n = len(fps)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    sim = similarity_matrix(fps, metric="tanimoto")
    dist = 1.0 - sim
    picked = [seed_idx % n]
    min_dist = dist[picked[0]].copy()

    while len(picked) < k:
        for idx in picked:
            min_dist[idx] = -1.0
        next_idx = int(np.argmax(min_dist))
        picked.append(next_idx)
        min_dist = np.minimum(min_dist, dist[next_idx])
    return picked


# ---------------------------------------------------------------------------
# Bemis-Murcko Scaffolds & Matched Molecular Pair (MMP) Fragmentation
# ---------------------------------------------------------------------------

def _subgraph_molecule(mol: MoleculeGraph, keep_indices: Sequence[int], generic: bool = False) -> MoleculeGraph:
    keep_set = set(keep_indices)
    remap = {old: new for new, old in enumerate(sorted(keep_set))}
    new_atoms: List[AtomRecord] = []
    for old in sorted(keep_set):
        orig = mol.atoms[old]
        new_atoms.append(
            AtomRecord(
                index=remap[old],
                element="C" if generic else orig.element,
                formal_charge=0 if generic else orig.formal_charge,
                is_aromatic=False if generic else orig.is_aromatic,
                explicit_h=0 if generic else None,
            )
        )
    new_bonds: List[BondRecord] = []
    for bond in mol.bonds:
        if bond.atom_u in keep_set and bond.atom_v in keep_set:
            u = remap[bond.atom_u]
            v = remap[bond.atom_v]
            new_bonds.append(
                BondRecord(
                    atom_u=min(u, v),
                    atom_v=max(u, v),
                    order=1.0 if generic else bond.order,
                    is_aromatic=False if generic else bond.is_aromatic,
                )
            )
    sub = MoleculeGraph(smiles="", atoms=new_atoms, bonds=new_bonds)
    if new_atoms:
        perceive_molecule(sub)
    return sub


def bemis_murcko_scaffold(mol: MoleculeGraph) -> ScaffoldDecomposition:
    """Extract Bemis-Murcko scaffold (rings + inter-ring linkers) and generic cyclic skeleton."""
    n = mol.num_atoms
    if n == 0 or not mol.sssr_rings:
        return ScaffoldDecomposition(
            original_smiles=mol.canonical_smiles or mol.smiles,
            murcko_smiles="",
            generic_skeleton_smiles="",
            ring_system_count=0,
            linker_atom_count=0,
            sidechain_atom_count=n,
        )

    # Iteratively prune terminal (degree <= 1) non-ring atoms
    adj: List[Set[int]] = [set(mol.neighbors(i)) for i in range(n)]
    alive = [True] * n
    ring_atoms = {i for i, a in enumerate(mol.atoms) if a.in_ring}

    changed = True
    while changed:
        changed = False
        for i in range(n):
            if alive[i] and i not in ring_atoms and len(adj[i]) <= 1:
                alive[i] = False
                for nbr in list(adj[i]):
                    adj[nbr].discard(i)
                adj[i].clear()
                changed = True

    kept = [i for i in range(n) if alive[i]]
    # Also preserve exocyclic carbonyl/imine double-bonded attachments to kept ring/linker?
    # Pure Bemis-Murcko removes all terminal sidechains (keeps rings + linkers strictly).
    murcko_mol = _subgraph_molecule(mol, kept, generic=False)
    generic_mol = _subgraph_molecule(mol, kept, generic=True)
    murcko_smi = canonical_smiles(murcko_mol) if kept else ""
    generic_smi = canonical_smiles(generic_mol) if kept else ""

    # Count distinct fused/spiro ring systems
    ring_adj: Dict[int, Set[int]] = {i: set() for i in range(len(mol.sssr_rings))}
    ring_sets = [set(cyc) for cyc in mol.sssr_rings]
    for i in range(len(ring_sets)):
        for j in range(i + 1, len(ring_sets)):
            if ring_sets[i] & ring_sets[j]:
                ring_adj[i].add(j)
                ring_adj[j].add(i)
    seen_r: Set[int] = set()
    ring_systems = 0
    for r in range(len(ring_sets)):
        if r not in seen_r:
            ring_systems += 1
            q = deque([r])
            seen_r.add(r)
            while q:
                cur = q.popleft()
                for nxt in ring_adj[cur]:
                    if nxt not in seen_r:
                        seen_r.add(nxt)
                        q.append(nxt)

    linker_count = sum(1 for i in kept if i not in ring_atoms)
    sidechain_count = n - len(kept)
    return ScaffoldDecomposition(
        original_smiles=mol.canonical_smiles or mol.smiles,
        murcko_smiles=murcko_smi,
        generic_skeleton_smiles=generic_smi,
        ring_system_count=ring_systems,
        linker_atom_count=linker_count,
        sidechain_atom_count=sidechain_count,
    )


def fragment_single_cut_mmp(mol: MoleculeGraph) -> List[Tuple[str, str]]:
    """Perform Hussain-Rea single-cut fragmentation over acyclic single bonds.

    Returns (core_smiles, substituent_smiles) pairs where core is the larger fragment.
    """
    n = mol.num_atoms
    if n < 3:
        return []

    adj: List[List[int]] = [mol.neighbors(i) for i in range(n)]
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for bond in mol.bonds:
        if bond.in_ring or bond.is_aromatic or bond.order >= 1.5:
            continue
        u, v = bond.atom_u, bond.atom_v
        # BFS from u avoiding edge (u, v)
        comp_u: List[int] = []
        visited = {u}
        q: deque[int] = deque([u])
        while q:
            cur = q.popleft()
            comp_u.append(cur)
            for nxt in adj[cur]:
                if (cur == u and nxt == v) or (cur == v and nxt == u):
                    continue
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        comp_v = [i for i in range(n) if i not in visited]
        if not comp_u or not comp_v:
            continue
        if len(comp_u) < len(comp_v):
            core_idx, frag_idx = comp_v, comp_u
        else:
            core_idx, frag_idx = comp_u, comp_v
        if len(frag_idx) > max(10, int(0.45 * n)):
            continue
        core_mol = _subgraph_molecule(mol, core_idx, generic=False)
        frag_mol = _subgraph_molecule(mol, frag_idx, generic=False)
        pair = (core_mol.canonical_smiles, frag_mol.canonical_smiles)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def mine_matched_molecular_pairs(
    smiles_list: Sequence[str],
    properties: Sequence[float],
    property_name: str = "pIC50",
) -> List[MMPTransform]:
    """Identify Matched Molecular Pairs sharing an identical Hussain-Rea core scaffold."""
    by_core: Dict[str, List[Tuple[str, str, float]]] = {}
    for smi, val in zip(smiles_list, properties):
        mol = parse_smiles(smi)
        for core_smi, frag_smi in fragment_single_cut_mmp(mol):
            by_core.setdefault(core_smi, []).append((frag_smi, mol.canonical_smiles, float(val)))

    transforms: List[MMPTransform] = []
    for core_smi, entries in by_core.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                f1, s1, v1 = entries[i]
                f2, s2, v2 = entries[j]
                if f1 == f2 or s1 == s2:
                    continue
                transforms.append(
                    MMPTransform(
                        core_smiles=core_smi,
                        from_fragment=f1,
                        to_fragment=f2,
                        from_smiles=s1,
                        to_smiles=s2,
                        delta_property=float(v2 - v1),
                        property_name=property_name,
                    )
                )
    transforms.sort(key=lambda t: abs(t.delta_property), reverse=True)
    return transforms
