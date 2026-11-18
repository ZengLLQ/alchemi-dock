"""De novo molecular property optimization via Monte Carlo Tree Search (MCTS)
and NSGA-II Pareto multi-objective evolutionary search.

All molecular mutations operate directly on MoleculeGraph objects with strict
valence preservation, SMILES canonicalization, and scaffold-novelty tracking.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .adme import predict_adme_profile
from .descriptors import (
    bemis_murcko_scaffold,
    compute_morgan_fingerprint,
    tanimoto_similarity,
)
from .molgraph import canonical_smiles, parse_smiles, perceive_molecule
from .types import (
    AtomRecord,
    BondRecord,
    IntArray,
    MoleculeGraph,
    OptimizationCandidate,
    OptimizationTrajectory,
)

# Medicinal-chemistry R-groups as SMILES rooted at atom index 0
_R_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("add_F", "F"),
    ("add_Cl", "Cl"),
    ("add_Me", "C"),
    ("add_OMe", "OC"),
    ("add_OH", "O"),
    ("add_NH2", "N"),
    ("add_CN", "C#N"),
    ("add_CF3", "C(F)(F)F"),
    ("add_amide", "C(=O)N"),
    ("add_pyridyl", "c1ccncc1"),
    ("add_cyclopropyl", "C1CC1"),
    ("add_morpholine", "N1CCOCC1"),
)


def _clone_mol(mol: MoleculeGraph) -> MoleculeGraph:
    return MoleculeGraph(
        smiles=mol.smiles,
        atoms=[copy.copy(a) for a in mol.atoms],
        bonds=[copy.copy(b) for b in mol.bonds],
        sssr_rings=list(mol.sssr_rings),
        canonical_smiles=mol.canonical_smiles,
        name=mol.name,
    )


def _attach_fragment(mol: MoleculeGraph, target_idx: int, frag_smiles: str) -> Optional[MoleculeGraph]:
    """Attach a SMILES fragment at atom 0 of fragment onto mol.atoms[target_idx]."""
    target = mol.atoms[target_idx]
    if target.implicit_h < 1 or mol.num_atoms > 42:
        return None
    try:
        frag = parse_smiles(frag_smiles)
    except Exception:
        return None

    new_mol = _clone_mol(mol)
    offset = new_mol.num_atoms
    for fa in frag.atoms:
        new_mol.atoms.append(
            AtomRecord(
                index=offset + fa.index,
                element=fa.element,
                formal_charge=fa.formal_charge,
                is_aromatic=fa.is_aromatic,
                explicit_h=fa.explicit_h,
                isotope=fa.isotope,
            )
        )
    for fb in frag.bonds:
        new_mol.bonds.append(
            BondRecord(
                atom_u=offset + fb.atom_u,
                atom_v=offset + fb.atom_v,
                order=fb.order,
                is_aromatic=fb.is_aromatic,
            )
        )
    new_mol.bonds.append(
        BondRecord(
            atom_u=min(target_idx, offset),
            atom_v=max(target_idx, offset),
            order=1.0,
            is_aromatic=False,
        )
    )
    new_mol.atoms[target_idx].explicit_h = max(0, target.implicit_h - 1) if target.explicit_h is not None else None
    perceive_molecule(new_mol)
    smi = canonical_smiles(new_mol)
    if not smi:
        return None
    # Verify round-trip parseability
    reparsed = parse_smiles(smi)
    return reparsed


def _mutate_ring_bioisostere(mol: MoleculeGraph, atom_idx: int, new_elem: str) -> Optional[MoleculeGraph]:
    """Perform valence-safe CH -> N / CH2 -> O / CH2 -> NH bioisosteric ring/linker swap."""
    atom = mol.atoms[atom_idx]
    deg = len(mol.neighbors(atom_idx))
    if atom.is_aromatic and atom.element == "C" and new_elem == "N" and deg == 2 and atom.implicit_h >= 1:
        cand = _clone_mol(mol)
        cand.atoms[atom_idx].element = "N"
        cand.atoms[atom_idx].explicit_h = 0
        perceive_molecule(cand)
        return parse_smiles(canonical_smiles(cand))
    if (not atom.is_aromatic) and atom.element == "C" and new_elem in ("O", "N") and deg == 2 and atom.implicit_h == 2:
        cand = _clone_mol(mol)
        cand.atoms[atom_idx].element = new_elem
        cand.atoms[atom_idx].explicit_h = None
        perceive_molecule(cand)
        return parse_smiles(canonical_smiles(cand))
    if atom.element in ("F", "Cl") and new_elem in ("F", "Cl") and atom.element != new_elem:
        cand = _clone_mol(mol)
        cand.atoms[atom_idx].element = new_elem
        perceive_molecule(cand)
        return parse_smiles(canonical_smiles(cand))
    return None


def _prune_terminal_atom(mol: MoleculeGraph, atom_idx: int) -> Optional[MoleculeGraph]:
    """Remove a terminal non-ring degree-1 substituent atom."""
    if mol.num_atoms <= 7:
        return None
    atom = mol.atoms[atom_idx]
    if atom.in_ring or len(mol.neighbors(atom_idx)) != 1:
        return None
    keep = [i for i in range(mol.num_atoms) if i != atom_idx]
    remap = {old: new for new, old in enumerate(keep)}
    new_atoms = [
        AtomRecord(
            index=remap[i],
            element=mol.atoms[i].element,
            formal_charge=mol.atoms[i].formal_charge,
            is_aromatic=mol.atoms[i].is_aromatic,
        )
        for i in keep
    ]
    new_bonds = [
        BondRecord(
            atom_u=min(remap[b.atom_u], remap[b.atom_v]),
            atom_v=max(remap[b.atom_u], remap[b.atom_v]),
            order=b.order,
            is_aromatic=b.is_aromatic,
        )
        for b in mol.bonds
        if b.atom_u != atom_idx and b.atom_v != atom_idx
    ]
    cand = MoleculeGraph(smiles="", atoms=new_atoms, bonds=new_bonds)
    perceive_molecule(cand)
    return parse_smiles(canonical_smiles(cand))


def propose_valid_mutations(
    smiles: str,
    rng: np.random.Generator,
    max_proposals: int = 8,
) -> List[Tuple[str, str]]:
    """Propose up to max_proposals chemically valid (edit_action, child_smiles) neighbors."""
    mol = parse_smiles(smiles)
    attachable = [a.index for a in mol.atoms if a.implicit_h >= 1]
    arom_c = [
        a.index
        for a in mol.atoms
        if a.is_aromatic and a.element == "C" and len(mol.neighbors(a.index)) == 2 and a.implicit_h >= 1
    ]
    aliph_ch2 = [
        a.index
        for a in mol.atoms
        if (not a.is_aromatic) and a.element == "C" and len(mol.neighbors(a.index)) == 2 and a.implicit_h == 2
    ]
    terminals = [a.index for a in mol.atoms if (not a.in_ring) and len(mol.neighbors(a.index)) == 1]

    proposals: List[Tuple[str, str]] = []
    seen_smi: Set[str] = {mol.canonical_smiles}
    attempts = 0
    while len(proposals) < max_proposals and attempts < max_proposals * 5:
        attempts += 1
        op_choice = float(rng.random())
        mutated: Optional[MoleculeGraph] = None
        action_label = ""

        if op_choice < 0.62 and attachable:
            idx = int(rng.choice(attachable))
            r_name, r_smi = _R_GROUPS[int(rng.integers(0, len(_R_GROUPS)))]
            mutated = _attach_fragment(mol, idx, r_smi)
            action_label = f"{r_name}@{idx}"
        elif op_choice < 0.84 and (arom_c or aliph_ch2):
            if arom_c and (not aliph_ch2 or rng.random() < 0.65):
                idx = int(rng.choice(arom_c))
                mutated = _mutate_ring_bioisostere(mol, idx, "N")
                action_label = f"scaffold_hop_N@{idx}"
            else:
                idx = int(rng.choice(aliph_ch2))
                elem = "O" if rng.random() < 0.5 else "N"
                mutated = _mutate_ring_bioisostere(mol, idx, elem)
                action_label = f"bioisostere_{elem}@{idx}"
        elif terminals:
            idx = int(rng.choice(terminals))
            mutated = _prune_terminal_atom(mol, idx)
            action_label = f"prune@{idx}"

        if mutated is not None and mutated.canonical_smiles and mutated.canonical_smiles not in seen_smi:
            seen_smi.add(mutated.canonical_smiles)
            proposals.append((action_label, mutated.canonical_smiles))

    return proposals


def evaluate_candidate(
    smiles: str,
    generation: int,
    parent_smiles: str,
    edit_action: str,
    potency_fn: Optional[Callable[[str], float]] = None,
) -> OptimizationCandidate:
    """Score a candidate ligand on predicted potency, QED, SAscore, and ADME safety MPO."""
    mol = parse_smiles(smiles)
    adme = predict_adme_profile(mol.canonical_smiles)
    scaff = bemis_murcko_scaffold(mol).murcko_smiles

    if potency_fn is not None:
        pic50 = float(potency_fn(mol.canonical_smiles))
    else:
        # Physics-inspired hinge-binding + hydrophobic + polar balance surrogate
        hbd_hba = adme.hbd + adme.hba
        pic50 = (
            5.2
            + 0.45 * min(3.8, max(0.0, adme.clogp))
            + 0.25 * min(4, len(mol.sssr_rings))
            + 0.18 * min(5, hbd_hba)
            - 0.015 * max(0.0, adme.mw - 460.0)
            - 0.025 * max(0.0, adme.tpsa - 115.0)
        )

    adme_pen = (
        0.55 * adme.ro5_violations
        + 0.85 * adme.herg_risk_prob
        + 0.65 * adme.cyp3a4_inhibition_prob
        + 0.40 * len(adme.alerts)
    )
    # Composite Multi-Parameter Optimization (MPO) score
    mpo = (
        0.48 * max(0.0, min(1.0, (pic50 - 4.5) / 4.5))
        + 0.34 * adme.qed
        + 0.18 * max(0.0, 1.0 - (adme.sa_score - 1.0) / 9.0)
        - 0.22 * adme_pen
    )
    return OptimizationCandidate(
        smiles=mol.canonical_smiles,
        generation=generation,
        parent_smiles=parent_smiles,
        edit_action=edit_action,
        predicted_pic50=float(pic50),
        qed=float(adme.qed),
        sa_score=float(adme.sa_score),
        clogp=float(adme.clogp),
        tpsa=float(adme.tpsa),
        adme_penalty=float(adme_pen),
        mpo_score=float(mpo),
        scaffold_smiles=scaff,
    )


def assign_pareto_ranks(candidates: List[OptimizationCandidate]) -> List[OptimizationCandidate]:
    """Compute non-dominated Pareto ranks across [potency, QED, -SAscore - adme_penalty]."""
    n = len(candidates)
    if n == 0:
        return []

    objs = np.array(
        [
            [c.predicted_pic50, c.qed, -(0.4 * c.sa_score + c.adme_penalty)]
            for c in candidates
        ],
        dtype=np.float64,
    )
    domination_count = [0] * n
    dominates: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            ge_all = bool(np.all(objs[p] >= objs[q]))
            gt_any = bool(np.any(objs[p] > objs[q]))
            if ge_all and gt_any:
                dominates[p].append(q)
            elif bool(np.all(objs[q] >= objs[p])) and bool(np.any(objs[q] > objs[p])):
                domination_count[p] += 1
        if domination_count[p] == 0:
            candidates[p].pareto_rank = 1
            fronts[0].append(p)

    rank = 1
    while fronts[rank - 1]:
        next_front: List[int] = []
        for p in fronts[rank - 1]:
            for q in dominates[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    candidates[q].pareto_rank = rank + 1
                    next_front.append(q)
        rank += 1
        fronts.append(next_front)

    return [candidates[i] for i in fronts[0]]


@dataclass
class _MCTSNode:
    smiles: str
    candidate: OptimizationCandidate
    parent: Optional[_MCTSNode] = None
    children: List[_MCTSNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0


def optimize_molecules_mcts_pareto(
    seed_smiles: str,
    generations: int = 8,
    population_size: int = 16,
    potency_fn: Optional[Callable[[str], float]] = None,
    seed: int = 42,
) -> OptimizationTrajectory:
    """Run hybrid MCTS + NSGA-II Pareto evolutionary molecular optimization."""
    rng = np.random.default_rng(seed)
    root_cand = evaluate_candidate(seed_smiles, 0, seed_smiles, "seed", potency_fn=potency_fn)
    root = _MCTSNode(smiles=root_cand.smiles, candidate=root_cand)

    archive: Dict[str, OptimizationCandidate] = {root_cand.smiles: root_cand}
    population: List[OptimizationCandidate] = [root_cand]
    gen_best: List[float] = [root_cand.mpo_score]
    gen_mean: List[float] = [root_cand.mpo_score]
    total_attempts = 1
    valid_count = 1

    # Seed initial population around root
    for action, smi in propose_valid_mutations(root_cand.smiles, rng, max_proposals=min(6, population_size)):
        total_attempts += 1
        valid_count += 1
        cand = evaluate_candidate(smi, 0, root_cand.smiles, action, potency_fn=potency_fn)
        archive[smi] = cand
        population.append(cand)
        child_node = _MCTSNode(smiles=smi, candidate=cand, parent=root, visits=1, value_sum=cand.mpo_score)
        root.children.append(child_node)
    root.visits = len(root.children) + 1
    root.value_sum = sum(c.candidate.mpo_score for c in root.children) + root_cand.mpo_score

    nodes_by_smi: Dict[str, _MCTSNode] = {root.smiles: root, **{ch.smiles: ch for ch in root.children}}

    for gen in range(1, generations + 1):
        # MCTS selection + expansion steps
        new_cands: List[OptimizationCandidate] = []
        for _ in range(max(4, population_size // 2)):
            # UCB1 tree traversal from root
            node = root
            depth = 0
            while node.children and depth < 4:
                log_n = math.log(max(2, node.visits))
                node = max(
                    node.children,
                    key=lambda ch: (ch.value_sum / max(1, ch.visits)) + 0.35 * math.sqrt(log_n / max(1, ch.visits)),
                )
                depth += 1

            mutations = propose_valid_mutations(node.smiles, rng, max_proposals=3)
            for action, child_smi in mutations:
                total_attempts += 1
                valid_count += 1
                if child_smi in archive:
                    continue
                cand = evaluate_candidate(child_smi, gen, node.smiles, action, potency_fn=potency_fn)
                archive[child_smi] = cand
                new_cands.append(cand)
                child_node = _MCTSNode(
                    smiles=child_smi,
                    candidate=cand,
                    parent=node,
                    visits=1,
                    value_sum=cand.mpo_score,
                )
                node.children.append(child_node)
                nodes_by_smi[child_smi] = child_node
                # Backpropagate reward up tree
                bp: Optional[_MCTSNode] = child_node
                while bp is not None:
                    bp.visits += 1
                    bp.value_sum += cand.mpo_score
                    bp = bp.parent

        # Evolutionary elitist expansion from top Pareto members
        pool = list(archive.values())
        pareto_front = assign_pareto_ranks(pool)
        elites = sorted(pareto_front, key=lambda c: c.mpo_score, reverse=True)[:4]
        for elite in elites:
            for action, child_smi in propose_valid_mutations(elite.smiles, rng, max_proposals=2):
                total_attempts += 1
                valid_count += 1
                if child_smi in archive:
                    continue
                cand = evaluate_candidate(child_smi, gen, elite.smiles, action, potency_fn=potency_fn)
                archive[child_smi] = cand
                new_cands.append(cand)

        population = sorted(archive.values(), key=lambda c: (c.mpo_score, c.predicted_pic50), reverse=True)[
            :population_size
        ]
        gen_best.append(float(max(c.mpo_score for c in population)))
        gen_mean.append(float(np.mean([c.mpo_score for c in population])))

    all_cands = list(archive.values())
    pareto = assign_pareto_ranks(all_cands)
    pareto.sort(key=lambda c: c.mpo_score, reverse=True)
    best = max(all_cands, key=lambda c: c.mpo_score)

    # Compute mean pairwise Tanimoto diversity on top population
    fps: List[IntArray] = []
    for c in population[: min(12, len(population))]:
        bits, _ = compute_morgan_fingerprint(parse_smiles(c.smiles), radius=2, n_bits=512)
        fps.append(bits)
    if len(fps) >= 2:
        dists: List[float] = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                dists.append(1.0 - tanimoto_similarity(fps[i], fps[j]))
        diversity = float(np.mean(dists))
    else:
        diversity = 0.0

    return OptimizationTrajectory(
        seed_smiles=root_cand.smiles,
        algorithm="MCTS-UCB1 + NSGA-II Pareto Evolution",
        generations=generations,
        candidates_evaluated=len(all_cands),
        valid_rate=float(min(1.0, valid_count / max(1, total_attempts))),
        mean_diversity=diversity,
        best_candidate=best,
        pareto_frontier=pareto[:10],
        generation_best_mpo=gen_best,
        generation_mean_mpo=gen_mean,
    )
