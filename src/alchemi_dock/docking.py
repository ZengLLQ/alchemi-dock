"""Structure-based empirical protein-ligand docking and interaction profiling.

Implements an AutoDock Vina-inspired scoring function combining:
- Steric Gauss1 and Gauss2 attractive interactions
- Quadratic steric clash repulsion
- Hydrophobic contact scoring
- Hydrogen-bond donor/acceptor interaction scoring
- Mehler-Solmajer distance-dependent screened electrostatics
- Torsional entropy penalty (N_rot)

Pose optimization combines rigid-body quaternion + translation + dihedral torsion
sampling via Monte Carlo Simulated Annealing followed by L-BFGS-B local refinement.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .molgraph import generate_conformer_3d, parse_smiles
from .types import (
    VDW_RADII,
    BindingPocket,
    Conformer3D,
    DockingPose,
    FloatArray,
    InteractionContact,
    MoleculeGraph,
    PocketAtom,
)


def build_kinase_atp_pocket() -> BindingPocket:
    """Construct a representative ATP-competitive kinase hinge/DFG binding pocket."""
    raw_atoms = [
        ("MET793", "N", "N", (-2.2, 1.4, 0.2), -0.35, True, False, False, False),
        ("MET793", "O", "O", (-2.6, -0.9, 0.4), -0.45, False, True, False, False),
        ("GLU762", "OE1", "O", (2.8, 2.2, 1.1), -0.55, False, True, False, False),
        ("LYS745", "NZ", "N", (2.4, -0.2, 2.1), 0.65, True, False, False, False),
        ("ASP855", "OD1", "O", (1.6, -2.5, -0.8), -0.55, False, True, False, False),
        ("THR790", "OG1", "O", (-0.6, 2.7, -1.2), -0.38, True, True, False, False),
        ("VAL726", "CG1", "C", (-0.4, -2.2, 2.4), 0.02, False, False, True, False),
        ("LEU718", "CD1", "C", (-3.1, -1.5, -2.1), 0.01, False, False, True, False),
        ("ALA743", "CB", "C", (0.2, 2.4, 2.1), 0.01, False, False, True, False),
        ("PHE856", "CZ", "C", (3.2, -1.8, -1.9), -0.05, False, False, True, True),
        ("PHE856", "CE1", "C", (2.4, -2.6, -2.5), -0.05, False, False, True, True),
        ("LEU844", "CD2", "C", (-1.1, 0.2, -2.9), 0.01, False, False, True, False),
        ("CYS797", "SG", "S", (-3.6, 1.8, 2.0), -0.15, True, True, True, False),
        ("ILE744", "CD1", "C", (1.3, 3.1, -0.4), 0.01, False, False, True, False),
    ]
    atoms: List[PocketAtom] = []
    for idx, (res, aname, elem, xyz, q, hbd, hba, hyd, arom) in enumerate(raw_atoms):
        atoms.append(
            PocketAtom(
                index=idx,
                residue_name=res,
                atom_name=aname,
                element=elem,
                coord=np.asarray(xyz, dtype=np.float64),
                partial_charge=q,
                is_hbd=hbd,
                is_hba=hba,
                is_hydrophobic=hyd,
                is_aromatic=arom,
            )
        )
    return BindingPocket(
        name="Kinase_ATP_Hinge_Pocket",
        target_family="Protein Kinase",
        center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        box_size=np.array([12.0, 12.0, 12.0], dtype=np.float64),
        atoms=atoms,
    )


def _quaternion_to_matrix(q: FloatArray) -> FloatArray:
    """Convert a quaternion [w, x, y, z] into a 3x3 proper rotation matrix."""
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rodrigues_rotate(points: FloatArray, origin: FloatArray, axis: FloatArray, angle: float) -> FloatArray:
    """Rotate 3D points around an axis through origin by angle (radians)."""
    ax_norm = float(np.linalg.norm(axis))
    if ax_norm < 1e-12 or abs(angle) < 1e-12:
        return points
    k = axis / ax_norm
    v = points - origin
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cross = np.cross(k, v)
    dot = np.sum(v * k, axis=1, keepdims=True)
    rotated = v * cos_a + cross * sin_a + k[None, :] * dot * (1.0 - cos_a)
    return rotated + origin


def _downstream_component(
    n_atoms: int,
    bonds: Sequence[Tuple[int, int, float]],
    u: int,
    v: int,
) -> List[int]:
    """Return atom indices on the v-side of rotatable bond (u, v)."""
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for a, b, _ in bonds:
        if (a == u and b == v) or (a == v and b == u):
            continue
        adj[a].append(b)
        adj[b].append(a)
    visited = {v}
    stack = [v]
    out: List[int] = []
    while stack:
        cur = stack.pop()
        out.append(cur)
        for nxt in adj[cur]:
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return out


def _ligand_atom_flags(
    mol: MoleculeGraph,
    conf: Conformer3D,
) -> Tuple[FloatArray, List[bool], List[bool], List[bool]]:
    """Return VDW radii and HBD/HBA/hydrophobic flags for heavy atoms of a conformer."""
    radii: List[float] = []
    is_hbd: List[bool] = []
    is_hba: List[bool] = []
    is_hyd: List[bool] = []

    for idx in conf.heavy_indices:
        atom = mol.atoms[idx]
        radii.append(VDW_RADII.get(atom.element, 1.70))
        hbd = atom.element in ("N", "O") and atom.implicit_h > 0
        hba = atom.element in ("N", "O")
        hyd = (atom.element == "C" and not any(mol.atoms[n].element in ("N", "O") for n in mol.neighbors(idx))) or (
            atom.element in ("F", "Cl", "Br", "I", "S")
        )
        is_hbd.append(hbd)
        is_hba.append(hba)
        is_hyd.append(hyd)

    return np.asarray(radii, dtype=np.float64), is_hbd, is_hba, is_hyd


def score_ligand_pose(
    mol: MoleculeGraph,
    ligand_heavy_coords: FloatArray,
    pocket: BindingPocket,
    heavy_charges: FloatArray,
    n_rotatable: int,
) -> Tuple[float, Dict[str, float], List[InteractionContact]]:
    """Compute Vina-inspired empirical binding score (kcal/mol) and interaction contacts."""
    n_lig = len(ligand_heavy_coords)
    n_pkt = len(pocket.atoms)
    if n_lig == 0 or n_pkt == 0:
        return 0.0, {}, []

    pkt_coords = np.vstack([a.coord for a in pocket.atoms])
    pkt_radii = np.array([VDW_RADII.get(a.element, 1.70) for a in pocket.atoms], dtype=np.float64)
    pkt_charges = np.array([a.partial_charge for a in pocket.atoms], dtype=np.float64)
    pkt_hbd = np.array([a.is_hbd for a in pocket.atoms], dtype=bool)
    pkt_hba = np.array([a.is_hba for a in pocket.atoms], dtype=bool)
    pkt_hyd = np.array([a.is_hydrophobic for a in pocket.atoms], dtype=bool)

    lig_radii = np.array([VDW_RADII.get(a.element, 1.70) for a in mol.atoms], dtype=np.float64)
    lig_hbd = np.array([a.element in ("N", "O") and a.implicit_h > 0 for a in mol.atoms], dtype=bool)
    lig_hba = np.array([a.element in ("N", "O") for a in mol.atoms], dtype=bool)
    lig_hyd = np.array(
        [
            (a.element == "C" and not any(mol.atoms[n].element in ("N", "O") for n in mol.neighbors(a.index)))
            or (a.element in ("F", "Cl", "Br", "I", "S"))
            for a in mol.atoms
        ],
        dtype=bool,
    )

    diff = ligand_heavy_coords[:, None, :] - pkt_coords[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    d = r - (lig_radii[:, None] + pkt_radii[None, :])  # surface distance

    # Cutoff mask within 8 A
    active = r < 8.0
    gauss1 = -0.035579 * float(np.sum(np.where(active, np.exp(-((d / 0.5) ** 2)), 0.0)))
    gauss2 = -0.005156 * float(np.sum(np.where(active, np.exp(-(((d - 3.0) / 2.0) ** 2)), 0.0)))
    repulsion = 0.840245 * float(np.sum(np.where(active & (d < 0.0), d ** 2, 0.0)))

    # Hydrophobic piecewise linear term
    hyd_mask = active & lig_hyd[:, None] & pkt_hyd[None, :]
    hyd_val = np.clip((1.5 - d) / 1.0, 0.0, 1.0)
    hydrophobic = -0.035069 * float(np.sum(np.where(hyd_mask, hyd_val, 0.0)))

    # Hydrogen bond term
    hb_pair = (lig_hbd[:, None] & pkt_hba[None, :]) | (lig_hba[:, None] & pkt_hbd[None, :])
    hb_mask = active & hb_pair
    hb_val = np.clip((-d) / 0.7, 0.0, 1.0)
    hbond = -0.587439 * float(np.sum(np.where(hb_mask, hb_val, 0.0)))

    # Mehler-Solmajer screened Coulomb electrostatics (kcal/mol)
    r_safe = np.maximum(1.2, r)
    u = 0.5 * r_safe
    eps_r = 78.4 - (78.4 - 4.0) * 0.5 * (u ** 2 + 2.0 * u + 2.0) * np.exp(-u)
    coulomb = 0.025 * 332.0637 * float(
        np.sum(np.where(active, (heavy_charges[:, None] * pkt_charges[None, :]) / (eps_r * r_safe), 0.0))
    )

    inter_score = gauss1 + gauss2 + repulsion + hydrophobic + hbond + coulomb
    torsion_penalty = 0.05846 * float(n_rotatable)
    # Vina-like conformational entropy scaling
    total_score = inter_score / (1.0 + 0.05846 * n_rotatable)

    # Extract notable contacts
    contacts: List[InteractionContact] = []
    for i in range(n_lig):
        for j in range(n_pkt):
            rij = float(r[i, j])
            if rij > 4.5:
                continue
            pa = pocket.atoms[j]
            la = mol.atoms[i]
            if hb_pair[i, j] and rij <= 3.5:
                contacts.append(
                    InteractionContact(
                        interaction_type="HBond",
                        ligand_atom_idx=i,
                        ligand_element=la.element,
                        residue_name=pa.residue_name,
                        pocket_atom_name=pa.atom_name,
                        distance_a=rij,
                        energy_kcal_mol=float(-0.587439 * max(0.0, min(1.0, -float(d[i, j]) / 0.7))),
                    )
                )
            elif lig_hyd[i] and pkt_hyd[j] and rij <= 4.2:
                contacts.append(
                    InteractionContact(
                        interaction_type="Hydrophobic",
                        ligand_atom_idx=i,
                        ligand_element=la.element,
                        residue_name=pa.residue_name,
                        pocket_atom_name=pa.atom_name,
                        distance_a=rij,
                        energy_kcal_mol=-0.035,
                    )
                )
            elif abs(heavy_charges[i] * pa.partial_charge) > 0.08 and rij <= 4.3:
                contacts.append(
                    InteractionContact(
                        interaction_type="Electrostatic",
                        ligand_atom_idx=i,
                        ligand_element=la.element,
                        residue_name=pa.residue_name,
                        pocket_atom_name=pa.atom_name,
                        distance_a=rij,
                        energy_kcal_mol=float(0.08 * 332.06 * heavy_charges[i] * pa.partial_charge / (15.0 * rij)),
                    )
                )

    contacts.sort(key=lambda c: c.energy_kcal_mol)
    terms = {
        "gauss1": gauss1,
        "gauss2": gauss2,
        "repulsion": repulsion,
        "hydrophobic": hydrophobic,
        "hbond": hbond,
        "electrostatic": coulomb,
        "torsion_penalty": torsion_penalty,
    }
    return float(total_score), terms, contacts[:16]


def _transform_conformer(
    base_heavy: FloatArray,
    translation: FloatArray,
    quat: FloatArray,
    torsions: FloatArray,
    rot_subtrees: Sequence[Tuple[int, int, List[int]]],
) -> FloatArray:
    coords = base_heavy.copy()
    for angle, (u, v, downstream) in zip(torsions, rot_subtrees):
        if downstream:
            coords[downstream] = _rodrigues_rotate(coords[downstream], coords[u], coords[v] - coords[u], float(angle))
    coords = coords - np.mean(coords, axis=0, keepdims=True)
    rot_m = _quaternion_to_matrix(quat)
    return coords @ rot_m.T + translation[None, :]


def dock_ligand(
    smiles: str,
    pocket: Optional[BindingPocket] = None,
    mc_steps: int = 160,
    seed: int = 42,
) -> DockingPose:
    """Dock a SMILES ligand into a BindingPocket using MC simulated annealing + L-BFGS-B."""
    if pocket is None:
        pocket = build_kinase_atp_pocket()

    mol = parse_smiles(smiles)
    conf = generate_conformer_3d(mol, seed=seed, add_hydrogens=False)
    base_heavy = conf.coords[conf.heavy_indices]
    heavy_charges = conf.charges[conf.heavy_indices]
    rot_bonds = conf.rotatable_bonds[:4]  # optimize up to 4 core torsions for fast convergence
    rot_subtrees = [(u, v, _downstream_component(mol.num_atoms, conf.bonds, u, v)) for u, v in rot_bonds]
    n_rot = len(conf.rotatable_bonds)

    rng = np.random.default_rng(seed)
    n_tors = len(rot_subtrees)

    def _eval_vec(vec: FloatArray) -> float:
        trans = vec[:3]
        quat = vec[3:7]
        tors = vec[7:]
        xyz = _transform_conformer(base_heavy, trans, quat, tors, rot_subtrees)
        box_pen = float(4.0 * np.sum(np.maximum(0.0, np.abs(trans - pocket.center) - pocket.box_size * 0.25) ** 2))
        score, _, _ = score_ligand_pose(mol, xyz, pocket, heavy_charges, n_rot)
        return score + box_pen

    best_vec = np.concatenate([pocket.center.copy(), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(n_tors)])
    best_score = _eval_vec(best_vec)
    cur_vec = best_vec.copy()
    cur_score = best_score

    # Simulated annealing MC search
    for step in range(mc_steps):
        temp = max(0.15, 2.5 * (1.0 - step / max(1, mc_steps)))
        cand = cur_vec.copy()
        cand[:3] += rng.normal(0.0, 0.65 * temp, size=3)
        cand[3:7] += rng.normal(0.0, 0.28 * temp, size=4)
        cand[3:7] /= max(1e-9, float(np.linalg.norm(cand[3:7])))
        if n_tors > 0:
            cand[7:] += rng.normal(0.0, 0.45 * temp, size=n_tors)

        cand_score = _eval_vec(cand)
        delta = cand_score - cur_score
        if delta < 0.0 or rng.random() < math.exp(-delta / max(0.05, temp)):
            cur_vec = cand
            cur_score = cand_score
            if cand_score < best_score:
                best_score = cand_score
                best_vec = cand.copy()

    # Local L-BFGS-B refinement
    res = minimize(_eval_vec, best_vec, method="L-BFGS-B", options={"maxiter": 60})
    opt_vec = res.x if res.fun <= best_score else best_vec

    opt_xyz = _transform_conformer(base_heavy, opt_vec[:3], opt_vec[3:7], opt_vec[7:], rot_subtrees)
    final_score, terms, contacts = score_ligand_pose(mol, opt_xyz, pocket, heavy_charges, n_rot)

    return DockingPose(
        smiles=mol.canonical_smiles or smiles,
        pocket_name=pocket.name,
        total_score_kcal_mol=final_score,
        gauss1_term=terms["gauss1"],
        gauss2_term=terms["gauss2"],
        repulsion_term=terms["repulsion"],
        hydrophobic_term=terms["hydrophobic"],
        hbond_term=terms["hbond"],
        electrostatic_term=terms["electrostatic"],
        torsion_penalty=terms["torsion_penalty"],
        ligand_coords=opt_xyz,
        contacts=contacts,
    )


def kabsch_rmsd(coords_a: FloatArray, coords_b: FloatArray) -> float:
    """Compute minimum heavy-atom RMSD (Angstrom) via optimal Kabsch SVD alignment."""
    if coords_a.shape != coords_b.shape or len(coords_a) == 0:
        return 0.0
    a = coords_a - np.mean(coords_a, axis=0, keepdims=True)
    b = coords_b - np.mean(coords_b, axis=0, keepdims=True)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    corr = np.diag([1.0, 1.0, -1.0 if d < 0.0 else 1.0])
    rot = vt.T @ corr @ u.T
    aligned = a @ rot.T
    return float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=1))))
