"""Pure-Python molecular graph engine: SMILES/SMARTS parsing, SSSR, Huckel aromaticity,
Kekulization, Gasteiger-Marsili partial charges, canonical SMILES, and 3D conformers.

No external chemistry toolkit is required. Every graph invariant (ring count E - V + C,
formal charge conservation, valence limits, and canonical round-trip stability) is
enforced deterministically.
"""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import minimize

from .types import (
    AROMATIC_ORGANIC_ELEMENTS,
    ATOMIC_MASSES,
    ATOMIC_NUMBERS,
    COVALENT_RADII,
    DEFAULT_VALENCES,
    GASTEIGER_PARAMS,
    ORGANIC_SUBSET,
    PAULING_EN,
    AtomRecord,
    BondRecord,
    Conformer3D,
    FloatArray,
    MoleculeGraph,
)

_BOND_SYMBOL_TO_ORDER: Dict[str, float] = {
    "-": 1.0,
    "=": 2.0,
    "#": 3.0,
    ":": 1.5,
    "/": 1.0,
    "\\": 1.0,
}

_BRACKET_RE = re.compile(
    r"^\[(?P<iso>\d+)?(?P<elem>[A-Za-z][a-z]?)(?P<chiral>@+)?(?P<h>H\d*)?(?P<chg>[+-]+\d*|[+-]*)?(?::(?P<map>\d+))?\]"
)


class SmilesParseError(ValueError):
    """Raised when a SMILES or SMARTS string is syntactically invalid."""


def _parse_bracket_atom(smiles: str, pos: int) -> Tuple[AtomRecord, int]:
    end = smiles.find("]", pos)
    if end == -1:
        raise SmilesParseError(f"Unclosed bracket atom at position {pos} in '{smiles}'")
    token = smiles[pos : end + 1]
    match = _BRACKET_RE.match(token)
    if not match:
        raise SmilesParseError(f"Invalid bracket atom '{token}' in '{smiles}'")

    iso_str = match.group("iso")
    raw_elem = match.group("elem")
    chiral = match.group("chiral")
    h_str = match.group("h")
    chg_str = match.group("chg")

    is_aromatic = raw_elem[0].islower() and raw_elem in AROMATIC_ORGANIC_ELEMENTS
    element = raw_elem.capitalize()
    if element not in ATOMIC_NUMBERS:
        # Allow arbitrary two-letter element fallback if valid capitalization
        element = raw_elem[0].upper() + raw_elem[1:].lower()

    explicit_h: Optional[int] = 0
    if h_str:
        explicit_h = 1 if len(h_str) == 1 else int(h_str[1:])

    formal_charge = 0
    if chg_str:
        if chg_str in ("+", "++", "+++"):
            formal_charge = len(chg_str)
        elif chg_str in ("-", "--", "---"):
            formal_charge = -len(chg_str)
        elif chg_str[0] == "+":
            formal_charge = int(chg_str[1:]) if len(chg_str) > 1 else 1
        elif chg_str[0] == "-":
            formal_charge = -int(chg_str[1:]) if len(chg_str) > 1 else -1

    atom = AtomRecord(
        index=0,
        element=element,
        formal_charge=formal_charge,
        is_aromatic=is_aromatic,
        explicit_h=explicit_h,
        isotope=int(iso_str) if iso_str else None,
        chirality=chiral,
    )
    return atom, end + 1


def _read_ring_number(smiles: str, pos: int) -> Tuple[int, int]:
    if smiles[pos] == "%":
        if pos + 2 >= len(smiles) or not smiles[pos + 1 : pos + 3].isdigit():
            raise SmilesParseError(f"Invalid ring closure '%..' at {pos} in '{smiles}'")
        return int(smiles[pos + 1 : pos + 3]), pos + 3
    if not smiles[pos].isdigit():
        raise SmilesParseError(f"Expected digit at {pos} in '{smiles}'")
    return int(smiles[pos]), pos + 1


def parse_smiles(smiles: str, name: str = "", perceive: bool = True) -> MoleculeGraph:
    """Parse a SMILES string into a fully perceived MoleculeGraph.

    Supports organic subset, aromatic lowercase atoms, bracket atoms with charges/isotopes/
    chirality/explicit hydrogens, single/double/triple/aromatic/stereo bonds, nested branches,
    ring closures (1-9 and %10-%99), and disconnected '.' fragments.
    """
    if not smiles or not isinstance(smiles, str):
        raise SmilesParseError("Empty or non-string SMILES")

    clean = smiles.strip()
    atoms: List[AtomRecord] = []
    bonds: List[BondRecord] = []
    stack: List[Optional[int]] = []
    ring_closures: Dict[int, Tuple[int, Optional[str]]] = {}

    prev_atom: Optional[int] = None
    pending_bond: Optional[str] = None
    pos = 0
    n = len(clean)

    while pos < n:
        ch = clean[pos]

        if ch in _BOND_SYMBOL_TO_ORDER:
            pending_bond = ch
            pos += 1
            continue

        if ch == ".":
            prev_atom = None
            pending_bond = None
            pos += 1
            continue

        if ch == "(":
            stack.append(prev_atom)
            pos += 1
            continue

        if ch == ")":
            if not stack:
                raise SmilesParseError(f"Unmatched ')' at position {pos} in '{clean}'")
            prev_atom = stack.pop()
            pending_bond = None
            pos += 1
            continue

        if ch.isdigit() or ch == "%":
            if prev_atom is None:
                raise SmilesParseError(f"Ring closure without preceding atom at {pos} in '{clean}'")
            rnum, pos = _read_ring_number(clean, pos)
            if rnum in ring_closures:
                open_atom, open_bond = ring_closures.pop(rnum)
                if open_atom == prev_atom:
                    raise SmilesParseError(f"Self-loop ring closure {rnum} in '{clean}'")
                bond_sym = pending_bond or open_bond
                if bond_sym is not None:
                    order = _BOND_SYMBOL_TO_ORDER[bond_sym]
                    is_arom = bond_sym == ":"
                elif atoms[open_atom].is_aromatic and atoms[prev_atom].is_aromatic:
                    order = 1.5
                    is_arom = True
                else:
                    order = 1.0
                    is_arom = False
                bonds.append(
                    BondRecord(
                        atom_u=min(open_atom, prev_atom),
                        atom_v=max(open_atom, prev_atom),
                        order=order,
                        is_aromatic=is_arom,
                        stereo=bond_sym if bond_sym in ("/", "\\") else None,
                    )
                )
                pending_bond = None
            else:
                ring_closures[rnum] = (prev_atom, pending_bond)
                pending_bond = None
            continue

        # Parse atom
        if ch == "[":
            atom, pos = _parse_bracket_atom(clean, pos)
        elif ch in AROMATIC_ORGANIC_ELEMENTS:
            atom = AtomRecord(index=0, element=ch.upper(), is_aromatic=True)
            pos += 1
        elif ch == "C" and pos + 1 < n and clean[pos + 1] == "l":
            atom = AtomRecord(index=0, element="Cl")
            pos += 2
        elif ch == "B" and pos + 1 < n and clean[pos + 1] == "r":
            atom = AtomRecord(index=0, element="Br")
            pos += 2
        elif ch in ORGANIC_SUBSET:
            atom = AtomRecord(index=0, element=ch)
            pos += 1
        else:
            raise SmilesParseError(f"Unexpected character '{ch}' at position {pos} in '{clean}'")

        atom.index = len(atoms)
        atoms.append(atom)

        if prev_atom is not None:
            if pending_bond is not None:
                order = _BOND_SYMBOL_TO_ORDER[pending_bond]
                is_arom = pending_bond == ":"
            elif atoms[prev_atom].is_aromatic and atom.is_aromatic:
                order = 1.5
                is_arom = True
            else:
                order = 1.0
                is_arom = False
            bonds.append(
                BondRecord(
                    atom_u=min(prev_atom, atom.index),
                    atom_v=max(prev_atom, atom.index),
                    order=order,
                    is_aromatic=is_arom,
                    stereo=pending_bond if pending_bond in ("/", "\\") else None,
                )
            )
            pending_bond = None

        prev_atom = atom.index

    if stack:
        raise SmilesParseError(f"Unclosed '(' branch in '{clean}'")
    if ring_closures:
        raise SmilesParseError(f"Unclosed ring closure(s) {sorted(ring_closures)} in '{clean}'")
    if not atoms:
        raise SmilesParseError(f"No atoms parsed from '{clean}'")

    mol = MoleculeGraph(smiles=clean, atoms=atoms, bonds=bonds, name=name)
    if perceive:
        perceive_molecule(mol)
    return mol


def _connected_components(n_atoms: int, adj: List[List[int]]) -> List[List[int]]:
    visited = [False] * n_atoms
    comps: List[List[int]] = []
    for start in range(n_atoms):
        if visited[start]:
            continue
        comp: List[int] = []
        queue: deque[int] = deque([start])
        visited[start] = True
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        comps.append(comp)
    return comps


def compute_sssr(mol: MoleculeGraph) -> List[Tuple[int, ...]]:
    """Compute the Smallest Set of Smallest Rings (SSSR) via Horton GF(2) minimal cycle basis.

    Returns exact E - V + C independent shortest cycles using bitwise Gaussian elimination
    over GF(2) on edge bitmasks.
    """
    n_atoms = mol.num_atoms
    if n_atoms == 0 or mol.num_bonds == 0:
        mol.sssr_rings = []
        return []

    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    edge_idx_map: Dict[Tuple[int, int], int] = {}
    for idx, bond in enumerate(mol.bonds):
        u, v = bond.atom_u, bond.atom_v
        adj[u].append(v)
        adj[v].append(u)
        edge_idx_map[(min(u, v), max(u, v))] = idx

    components = _connected_components(n_atoms, adj)
    expected_cycles = sum(
        sum(len(adj[u]) for u in comp) // 2 - len(comp) + 1 for comp in components
    )
    if expected_cycles <= 0:
        mol.sssr_rings = []
        for atom in mol.atoms:
            atom.in_ring = False
            atom.ring_sizes = ()
        for bond in mol.bonds:
            bond.in_ring = False
        return []

    # Shortest paths from every vertex via BFS
    parent_map: List[Dict[int, int]] = []
    dist_map: List[Dict[int, int]] = []
    for root in range(n_atoms):
        parents: Dict[int, int] = {root: -1}
        dists: Dict[int, int] = {root: 0}
        q: deque[int] = deque([root])
        while q:
            curr = q.popleft()
            for nbr in adj[curr]:
                if nbr not in dists:
                    dists[nbr] = dists[curr] + 1
                    parents[nbr] = curr
                    q.append(nbr)
        parent_map.append(parents)
        dist_map.append(dists)

    def _path_to_root(root: int, node: int) -> List[int]:
        path: List[int] = []
        curr = node
        parents = parent_map[root]
        while curr != -1:
            path.append(curr)
            curr = parents[curr]
        path.reverse()
        return path

    # Generate Horton candidate cycles
    seen_masks: Set[int] = set()
    candidates: List[Tuple[int, int, Tuple[int, ...]]] = []

    for bond in mol.bonds:
        x, y = bond.atom_u, bond.atom_v
        for root in range(n_atoms):
            if x not in dist_map[root] or y not in dist_map[root]:
                continue
            px = _path_to_root(root, x)
            py = _path_to_root(root, y)
            if set(px) & set(py) != {root}:
                continue
            # Ordered cycle: root -> ... -> x -> y -> ... -> root
            cyc_nodes = px + py[-1:0:-1]
            if len(cyc_nodes) < 3:
                continue
            mask = 0
            for k in range(len(cyc_nodes)):
                a = cyc_nodes[k]
                b = cyc_nodes[(k + 1) % len(cyc_nodes)]
                e_idx = edge_idx_map[(min(a, b), max(a, b))]
                mask |= 1 << e_idx
            if mask not in seen_masks:
                seen_masks.add(mask)
                candidates.append((len(cyc_nodes), mask, tuple(cyc_nodes)))

    candidates.sort(key=lambda item: (item[0], item[1]))

    # GF(2) Gaussian elimination to pick shortest independent cycle basis
    basis: Dict[int, int] = {}
    sssr: List[Tuple[int, ...]] = []
    for _, mask, cyc_tup in candidates:
        m = mask
        while m:
            pivot = m.bit_length() - 1
            if pivot in basis:
                m ^= basis[pivot]
            else:
                basis[pivot] = m
                sssr.append(cyc_tup)
                break
        if len(sssr) == expected_cycles:
            break

    mol.sssr_rings = sssr

    atom_rings: List[List[int]] = [[] for _ in range(n_atoms)]
    ring_edge_set: Set[Tuple[int, int]] = set()
    for cyc in sssr:
        rlen = len(cyc)
        for k, a in enumerate(cyc):
            atom_rings[a].append(rlen)
            b = cyc[(k + 1) % rlen]
            ring_edge_set.add((min(a, b), max(a, b)))

    for idx, atom in enumerate(mol.atoms):
        atom.in_ring = len(atom_rings[idx]) > 0
        atom.ring_sizes = tuple(sorted(atom_rings[idx]))

    for bond in mol.bonds:
        bond.in_ring = (min(bond.atom_u, bond.atom_v), max(bond.atom_u, bond.atom_v)) in ring_edge_set

    return sssr


def _update_implicit_hydrogens(mol: MoleculeGraph) -> None:
    """Compute implicit hydrogens and hybridization for all heavy atoms."""
    for atom in mol.atoms:
        nbr_bonds = mol.neighbor_bonds(atom.index)
        bond_order_sum = sum(b.order for _, b in nbr_bonds)
        has_triple = any(b.order >= 2.5 for _, b in nbr_bonds)
        has_double = any(1.75 <= b.order < 2.5 for _, b in nbr_bonds)
        has_arom = atom.is_aromatic or any(b.is_aromatic for _, b in nbr_bonds)

        if has_triple or (sum(1 for _, b in nbr_bonds if 1.75 <= b.order < 2.5) >= 2):
            atom.hybridization = "sp"
        elif has_double or has_arom:
            atom.hybridization = "sp2"
        else:
            atom.hybridization = "sp3"

        if atom.explicit_h is not None:
            atom.implicit_h = atom.explicit_h
            continue

        elem = atom.element
        if atom.is_aromatic:
            # Aromatic SMILES implicit hydrogen rules
            arom_deg = sum(1 for _, b in nbr_bonds if b.is_aromatic)
            exo_order = sum(b.order for _, b in nbr_bonds if not b.is_aromatic)
            total_used = arom_deg + exo_order + (1 if atom.element == "C" else 0)
            if elem == "C":
                # sp2 carbon in aromatic ring contributes 1 pi electron (double bond equivalent)
                atom.implicit_h = max(0, int(round(4.0 + atom.formal_charge - (arom_deg + 1.0 + exo_order))))
            elif elem == "N":
                if atom.formal_charge == 1:
                    atom.implicit_h = max(0, int(round(4.0 - (arom_deg + 1.0 + exo_order))))
                else:
                    # Pyridine-like N has 0 implicit H unless explicitly bracketed [nH]
                    atom.implicit_h = 0
            elif elem in ("O", "S"):
                atom.implicit_h = 0
            else:
                atom.implicit_h = max(0, int(round(3.0 - total_used)))
            continue

        valences = DEFAULT_VALENCES.get(elem, (4,))
        if elem in ("N", "P", "S", "O"):
            eff_val = bond_order_sum - atom.formal_charge
        else:
            eff_val = bond_order_sum + abs(atom.formal_charge)
        if elem == "C" and atom.formal_charge != 0:
            target = max(0, 4 - abs(atom.formal_charge))
            atom.implicit_h = max(0, int(round(target - bond_order_sum)))
        elif elem == "N" and atom.formal_charge == 1:
            atom.implicit_h = max(0, int(round(4.0 - bond_order_sum)))
        elif elem == "O" and atom.formal_charge == -1:
            atom.implicit_h = 0
        else:
            chosen = valences[-1]
            for val in valences:
                if val >= eff_val - 1e-6:
                    chosen = val
                    break
            if atom.formal_charge > 0:
                used = bond_order_sum - atom.formal_charge
            else:
                used = bond_order_sum + abs(atom.formal_charge)
            if elem == "N" and atom.formal_charge < 0:
                used = bond_order_sum + abs(atom.formal_charge)
            atom.implicit_h = max(0, int(round(chosen - used)))


def _pi_electron_contribution(mol: MoleculeGraph, atom_idx: int) -> Optional[int]:
    """Return pi-electron count contributed by atom_idx to a conjugated ring, or None if sp3."""
    atom = mol.atoms[atom_idx]
    nbr_bonds = mol.neighbor_bonds(atom_idx)
    exo_double_to_hetero = any(
        not b.in_ring and b.order >= 1.8 and mol.atoms[nbr].element in ("O", "N", "S")
        for nbr, b in nbr_bonds
    )
    in_ring_double = any(b.in_ring and (b.order >= 1.8 or b.is_aromatic) for _, b in nbr_bonds)

    if atom.element == "C":
        if exo_double_to_hetero:
            return 0  # Carbonyl / imine carbon acts as empty p-orbital (0 pi electrons)
        if atom.is_aromatic or in_ring_double or atom.formal_charge != 0:
            if atom.formal_charge == 1:
                return 0
            if atom.formal_charge == -1:
                return 2
            return 1
        return None
    if atom.element == "N":
        total_h = atom.implicit_h
        if atom.is_aromatic:
            if total_h > 0 or len(nbr_bonds) == 3:
                return 2  # Pyrrole-like NH / tertiary aromatic N
            return 1  # Pyridine-like N
        if in_ring_double:
            return 1
        if total_h > 0 or len(nbr_bonds) == 3:
            return 2
        return None
    if atom.element in ("O", "S"):
        if atom.is_aromatic or len(nbr_bonds) == 2:
            return 2
        return None
    if atom.element == "B":
        return 0
    return None


def perceive_aromaticity(mol: MoleculeGraph) -> None:
    """Apply Huckel (4n+2) aromaticity perception across SSSR rings and fused systems."""
    if not mol.sssr_rings:
        return

    for cyc in mol.sssr_rings:
        if len(cyc) < 5 or len(cyc) > 7:
            continue
        # Already marked aromatic from SMILES or check Kekule 4n+2
        pi_sum = 0
        all_planar = True
        for a_idx in cyc:
            contrib = _pi_electron_contribution(mol, a_idx)
            if contrib is None:
                all_planar = False
                break
            pi_sum += contrib
        if all_planar and pi_sum >= 2 and (pi_sum - 2) % 4 == 0:
            for k, u in enumerate(cyc):
                mol.atoms[u].is_aromatic = True
                v = cyc[(k + 1) % len(cyc)]
                bond = mol.get_bond(u, v)
                if bond is not None:
                    bond.is_aromatic = True
                    bond.is_conjugated = True


def assign_kekule_bonds(mol: MoleculeGraph) -> bool:
    """Assign alternating single (1.0) and double (2.0) Kekule bond orders to aromatic bonds.

    Uses maximum cardinality matching on aromatic atoms needing one pi double bond.
    Returns True if Kekulization succeeds for all aromatic atoms requiring a pi bond.
    """
    arom_bonds = [b for b in mol.bonds if b.is_aromatic]
    if not arom_bonds:
        return True

    needs_double: Set[int] = set()
    for atom in mol.atoms:
        if not atom.is_aromatic:
            continue
        nbr_bonds = mol.neighbor_bonds(atom.index)
        has_exo_double = any(not b.is_aromatic and b.order >= 1.8 for _, b in nbr_bonds)
        if has_exo_double:
            continue
        if atom.element == "C" and atom.formal_charge == 0:
            needs_double.add(atom.index)
        elif atom.element == "N" and atom.implicit_h == 0 and atom.formal_charge == 0 and len(nbr_bonds) <= 2:
            needs_double.add(atom.index)
        elif atom.element == "N" and atom.formal_charge == 1 and atom.implicit_h == 0:
            needs_double.add(atom.index)

    if not needs_double:
        for b in arom_bonds:
            b.order = 1.0
        return True

    # Adjacency restricted to aromatic bonds between atoms needing a double bond
    arom_adj: Dict[int, List[int]] = {u: [] for u in needs_double}
    for b in arom_bonds:
        if b.atom_u in needs_double and b.atom_v in needs_double:
            arom_adj[b.atom_u].append(b.atom_v)
            arom_adj[b.atom_v].append(b.atom_u)

    # Edmonds Blossom-like / randomized augmenting-path maximum matching
    match: Dict[int, int] = {}

    def _find_aug(start: int) -> bool:
        parent: Dict[int, Optional[int]] = {start: None}
        q: deque[int] = deque([start])
        visited_odd: Set[int] = set()
        while q:
            u = q.popleft()
            for v in arom_adj[u]:
                if v in visited_odd:
                    continue
                if v not in match:
                    # Augment along path
                    curr: Optional[int] = u
                    nxt = v
                    while curr is not None:
                        prev_mate = match.get(curr)
                        match[curr] = nxt
                        match[nxt] = curr
                        nxt = prev_mate if prev_mate is not None else -1
                        curr = parent[curr]
                    return True
                mate = match[v]
                if mate != u and mate not in parent:
                    visited_odd.add(v)
                    parent[mate] = u
                    q.append(mate)
        return False

    for u in sorted(needs_double):
        if u not in match:
            _find_aug(u)

    # Apply Kekule orders to aromatic bonds
    matched_edges: Set[Tuple[int, int]] = set()
    for u, v in match.items():
        if u < v:
            matched_edges.add((u, v))

    for b in arom_bonds:
        pair = (min(b.atom_u, b.atom_v), max(b.atom_u, b.atom_v))
        b.order = 2.0 if pair in matched_edges else 1.0
        b.is_conjugated = True

    return len(match) == len(needs_double)


def compute_gasteiger_charges(mol: MoleculeGraph, iterations: int = 6) -> FloatArray:
    """Compute Gasteiger-Marsili partial atomic charges over the molecular graph.

    Iterates damped electronegativity equalization across covalent bonds (damping 0.5^k).
    Strictly preserves total molecular formal charge: sum(q_i) == sum(formal_charge_i).
    """
    n = mol.num_atoms
    charges = np.array([float(a.formal_charge) for a in mol.atoms], dtype=np.float64)
    if n == 0:
        return charges

    # Pre-cache orbital polynomial parameters (a, b, c) and saturation chi_plus
    abc: List[Tuple[float, float, float]] = []
    chi_plus: List[float] = []
    for atom in mol.atoms:
        key = (atom.element, atom.hybridization)
        if key not in GASTEIGER_PARAMS:
            key = (atom.element, "sp3")
        a, b, c = GASTEIGER_PARAMS.get(key, (8.0, 9.0, 1.5))
        abc.append((a, b, c))
        chi_plus.append(a + b + c)

    h_a, h_b, h_c = GASTEIGER_PARAMS[("H", "sp3")]
    h_chi_plus = h_a + h_b + h_c
    h_charges = [0.0 for _ in mol.atoms]  # Aggregate implicit H charge per heavy atom

    for it in range(1, iterations + 1):
        damping = 0.5 ** it
        chi = [abc[i][0] + abc[i][1] * charges[i] + abc[i][2] * (charges[i] ** 2) for i in range(n)]
        delta = np.zeros(n, dtype=np.float64)

        # Heavy-heavy bonds
        for bond in mol.bonds:
            u, v = bond.atom_u, bond.atom_v
            if chi[v] > chi[u]:
                denom = chi_plus[u] if abs(chi_plus[u]) > 1e-9 else 1.0
                dq = (chi[v] - chi[u]) / denom * damping
            else:
                denom = chi_plus[v] if abs(chi_plus[v]) > 1e-9 else 1.0
                dq = (chi[v] - chi[u]) / denom * damping
            delta[u] += dq
            delta[v] -= dq

        # Implicit hydrogens attached to each heavy atom
        for i, atom in enumerate(mol.atoms):
            if atom.implicit_h <= 0:
                continue
            q_h_single = h_charges[i] / atom.implicit_h
            chi_h = h_a + h_b * q_h_single + h_c * (q_h_single ** 2)
            if chi[i] > chi_h:
                dq_h = (chi[i] - chi_h) / h_chi_plus * damping
            else:
                denom = chi_plus[i] if abs(chi_plus[i]) > 1e-9 else 1.0
                dq_h = (chi[i] - chi_h) / denom * damping
            total_dq = dq_h * atom.implicit_h
            h_charges[i] += total_dq
            delta[i] -= total_dq

        charges += delta

    # Fold implicit H charges back into net heavy-atom dipole representation if requested,
    # or return exact heavy+implicit-H balanced heavy atom net charges preserving total formal charge.
    # Here we return heavy-atom partial charges including attached implicit H net polarization
    # folded onto heavy atoms? Wait: standard Gasteiger keeps H separate, so heavy + H sums to formal charge.
    # To ensure sum(atom.partial_charge) == sum(formal_charge) on the heavy-atom graph itself,
    # we keep heavy-atom net-group charge OR distribute residual cleanly while also tracking raw charges.
    # Actually: let's add attached implicit H charge into a united-atom group charge ONLY if H not explicit,
    # Wait — if we fold h_charges into charges, sum(charges) == sum(formal_charge) identically!
    united_charges = charges + np.array(h_charges, dtype=np.float64)
    for idx, atom in enumerate(mol.atoms):
        atom.partial_charge = float(united_charges[idx])
    return united_charges


def canonical_smiles(mol: MoleculeGraph) -> str:
    """Generate a deterministic canonical SMILES string via Morgan-invariant symmetry breaking."""
    n = mol.num_atoms
    if n == 0:
        return ""

    # Compute initial topological invariants
    invariants: List[Tuple[int, int, int, int, int]] = []
    for a in mol.atoms:
        deg = len(mol.neighbors(a.index))
        invariants.append(
            (
                ATOMIC_NUMBERS.get(a.element, 6),
                deg,
                a.implicit_h,
                a.formal_charge,
                1 if a.is_aromatic else 0,
            )
        )

    # Iterative Morgan rank refinement
    ranks = _compress_ranks(invariants)
    for _ in range(max(4, n)):
        sigs: List[Tuple[int, Tuple[int, ...]]] = []
        for idx in range(n):
            nbr_ranks = sorted(ranks[nbr] for nbr in mol.neighbors(idx))
            sigs.append((ranks[idx], tuple(nbr_ranks)))
        new_ranks = _compress_ranks(sigs)
        if new_ranks == ranks:
            break
        ranks = new_ranks

    # Break any remaining symmetry ties deterministically by current index
    adj_sorted: List[List[int]] = []
    for idx in range(n):
        nbrs = sorted(mol.neighbors(idx), key=lambda nb: (ranks[nb], nb))
        adj_sorted.append(nbrs)

    visited = [False] * n
    fragments: List[str] = []

    # DFS spanning tree + ring closure assignment per connected component
    components = _connected_components(n, adj_sorted)
    components.sort(key=lambda c: min(ranks[u] for u in c))

    for comp in components:
        start = min(comp, key=lambda u: (ranks[u], u))
        ring_counter = 1
        back_edges: Dict[int, List[Tuple[int, int, BondRecord]]] = {u: [] for u in comp}
        dfs_order: List[int] = []
        tree_children: Dict[int, List[int]] = {u: [] for u in comp}
        parent_of: Dict[int, Optional[int]] = {start: None}
        seen_back: Set[Tuple[int, int]] = set()

        def _dfs(u: int) -> None:
            nonlocal ring_counter
            visited[u] = True
            dfs_order.append(u)
            for v in adj_sorted[u]:
                if not visited[v]:
                    parent_of[v] = u
                    tree_children[u].append(v)
                    _dfs(v)
                elif v != parent_of.get(u):
                    pair = (min(u, v), max(u, v))
                    if pair not in seen_back:
                        seen_back.add(pair)
                        bond = mol.get_bond(u, v)
                        assert bond is not None
                        back_edges[u].append((ring_counter, v, bond))
                        back_edges[v].append((ring_counter, u, bond))
                        ring_counter += 1

        _dfs(start)

        def _format_atom(u: int) -> str:
            a = mol.atoms[u]
            need_bracket = (
                a.formal_charge != 0
                or a.isotope is not None
                or a.element not in ORGANIC_SUBSET
                or (a.is_aromatic and a.element == "N" and a.implicit_h > 0)
            )
            sym = a.element.lower() if a.is_aromatic else a.element
            if not need_bracket:
                return sym
            parts = ["["]
            if a.isotope is not None:
                parts.append(str(a.isotope))
            parts.append(sym)
            if a.implicit_h > 0:
                parts.append("H" if a.implicit_h == 1 else f"H{a.implicit_h}")
            if a.formal_charge > 0:
                parts.append("+" if a.formal_charge == 1 else f"+{a.formal_charge}")
            elif a.formal_charge < 0:
                parts.append("-" if a.formal_charge == -1 else f"{a.formal_charge}")
            parts.append("]")
            return "".join(parts)

        def _format_bond(u: int, v: int, bond: BondRecord) -> str:
            if bond.is_aromatic or (mol.atoms[u].is_aromatic and mol.atoms[v].is_aromatic):
                return ""
            if bond.order >= 2.8:
                return "#"
            if bond.order >= 1.8:
                return "="
            return ""

        def _ring_token(num: int) -> str:
            return str(num) if num < 10 else f"%{num}"

        def _emit(u: int) -> str:
            out = [_format_atom(u)]
            for rnum, other, b in back_edges[u]:
                # Emit bond symbol on the opening side only
                if dfs_order.index(u) < dfs_order.index(other):
                    out.append(_format_bond(u, other, b) + _ring_token(rnum))
                else:
                    out.append(_ring_token(rnum))

            children = tree_children[u]
            if not children:
                return "".join(out)
            for child in children[:-1]:
                cbond = mol.get_bond(u, child)
                assert cbond is not None
                out.append("(" + _format_bond(u, child, cbond) + _emit(child) + ")")
            last = children[-1]
            lbond = mol.get_bond(u, last)
            assert lbond is not None
            out.append(_format_bond(u, last, lbond) + _emit(last))
            return "".join(out)

        fragments.append(_emit(start))

    canon = ".".join(sorted(fragments))
    mol.canonical_smiles = canon
    return canon


def _compress_ranks(items: Sequence[object]) -> List[int]:
    order = sorted(set(items), key=repr)
    mapping = {val: i for i, val in enumerate(order)}
    return [mapping[val] for val in items]


def perceive_molecule(mol: MoleculeGraph) -> MoleculeGraph:
    """Run full molecular graph perception: SSSR, Huckel aromaticity, Kekulization,
    valence/implicit H, Gasteiger charges, and canonical SMILES.
    """
    compute_sssr(mol)
    _update_implicit_hydrogens(mol)
    perceive_aromaticity(mol)
    assign_kekule_bonds(mol)
    _update_implicit_hydrogens(mol)
    compute_gasteiger_charges(mol)
    canonical_smiles(mol)
    return mol


# ---------------------------------------------------------------------------
# SMARTS Substructure Matcher (Backtracking Subgraph Isomorphism)
# ---------------------------------------------------------------------------

def _parse_smarts_atom_expr(expr: str) -> List[Tuple[str, object]]:
    """Parse an atom expression inside [...] or bare SMARTS atom."""
    clauses: List[Tuple[str, object]] = []
    pos = 0
    n = len(expr)
    while pos < n:
        neg = False
        if expr[pos] == "!":
            neg = True
            pos += 1
        if pos >= n:
            break
        ch = expr[pos]
        if ch == "#":
            m = re.match(r"#(\d+)", expr[pos:])
            if m:
                clauses.append(("atomic_num" if not neg else "not_atomic_num", int(m.group(1))))
                pos += len(m.group(0))
                continue
        if ch in ("D", "X", "H", "R", "r"):
            m = re.match(r"([DXHRr])(\d*)", expr[pos:])
            if m:
                key = m.group(1)
                val = int(m.group(2)) if m.group(2) else None
                clauses.append((f"prop_{key}_{neg}", val))
                pos += len(m.group(0))
                continue
        if ch in ("+", "-"):
            m = re.match(r"([+-])(\d*)", expr[pos:])
            if m:
                sign = 1 if m.group(1) == "+" else -1
                mag = int(m.group(2)) if m.group(2) else 1
                clauses.append(("charge", sign * mag))
                pos += len(m.group(0))
                continue
        if ch == "a":
            clauses.append(("aromatic", not neg))
            pos += 1
            continue
        if ch == "A":
            clauses.append(("aromatic", neg))
            pos += 1
            continue
        if ch == "*":
            clauses.append(("any", True))
            pos += 1
            continue
        if ch in (";", "&", ","):
            pos += 1
            continue
        if ch.isalpha():
            if pos + 1 < n and expr[pos : pos + 2] in ("Cl", "Br"):
                sym = expr[pos : pos + 2]
                pos += 2
            else:
                sym = ch
                pos += 1
            clauses.append(("elem", (sym, neg)))
            continue
        pos += 1
    return clauses


def _match_atom_expr(mol: MoleculeGraph, atom_idx: int, expr: str) -> bool:
    """Evaluate a SMARTS atom expression against mol.atoms[atom_idx]."""
    atom = mol.atoms[atom_idx]
    # Handle comma-OR groups first
    or_branches = expr.split(",")
    for branch in or_branches:
        and_parts = [p for p in re.split(r"[;&]", branch) if p]
        branch_ok = True
        for part in and_parts:
            clauses = _parse_smarts_atom_expr(part)
            for kind, val in clauses:
                if kind == "any":
                    continue
                if kind == "atomic_num":
                    if ATOMIC_NUMBERS.get(atom.element, 0) != val:
                        branch_ok = False
                elif kind == "not_atomic_num":
                    if ATOMIC_NUMBERS.get(atom.element, 0) == val:
                        branch_ok = False
                elif kind == "charge":
                    if atom.formal_charge != val:
                        branch_ok = False
                elif kind == "aromatic":
                    if atom.is_aromatic != bool(val):
                        branch_ok = False
                elif kind == "elem":
                    elem_pair = val
                    assert isinstance(elem_pair, tuple)
                    sym = str(elem_pair[0])
                    neg_elem = bool(elem_pair[1])
                    is_arom_req = sym.islower() and sym in AROMATIC_ORGANIC_ELEMENTS
                    req_elem = sym.capitalize() if is_arom_req else sym
                    matched = atom.element == req_elem and (
                        atom.is_aromatic if is_arom_req else (not atom.is_aromatic if sym in ORGANIC_SUBSET else True)
                    )
                    if sym in ("Cl", "Br", "F", "I", "P", "S", "B", "N", "O", "C") and not is_arom_req:
                        matched = atom.element == req_elem and not atom.is_aromatic
                    if neg_elem:
                        matched = not matched
                    if not matched:
                        branch_ok = False
                elif isinstance(kind, str) and kind.startswith("prop_"):
                    _, prop, neg_str = kind.split("_")
                    neg_flag = neg_str == "True"
                    deg = len(mol.neighbors(atom_idx))
                    total_deg = deg + atom.implicit_h
                    if prop == "D":
                        ok = deg == (val if val is not None else 1)
                    elif prop == "X":
                        ok = total_deg == (val if val is not None else 1)
                    elif prop == "H":
                        ok = atom.implicit_h == (val if val is not None else 1)
                    elif prop == "r":
                        ok = atom.in_ring if val is None else (val in atom.ring_sizes)
                    elif prop == "R":
                        ok = len(atom.ring_sizes) > 0 if val is None else (len(atom.ring_sizes) == val)
                    else:
                        ok = False
                    if neg_flag:
                        ok = not ok
                    if not ok:
                        branch_ok = False
        if branch_ok:
            return True
    return False


def _parse_smarts_query(smarts: str) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """Parse a SMARTS string into query atom expressions and bond patterns."""
    q_atoms: List[str] = []
    q_bonds: List[Tuple[int, int, str]] = []
    stack: List[Optional[int]] = []
    rings: Dict[int, Tuple[int, str]] = {}
    prev: Optional[int] = None
    pending_bond = ""
    pos = 0
    n = len(smarts)

    while pos < n:
        ch = smarts[pos]
        if ch in ("-", "=", "#", ":", "~", "@"):
            pending_bond = ch
            pos += 1
            continue
        if ch == "(":
            stack.append(prev)
            pos += 1
            continue
        if ch == ")":
            prev = stack.pop()
            pending_bond = ""
            pos += 1
            continue
        if ch.isdigit():
            rnum = int(ch)
            assert prev is not None
            if rnum in rings:
                u, obond = rings.pop(rnum)
                q_bonds.append((u, prev, pending_bond or obond))
                pending_bond = ""
            else:
                rings[rnum] = (prev, pending_bond)
                pending_bond = ""
            pos += 1
            continue
        if ch == "[":
            end = smarts.find("]", pos)
            if end == -1:
                raise SmilesParseError(f"Unclosed SMARTS '[' in '{smarts}'")
            expr = smarts[pos + 1 : end]
            pos = end + 1
        elif smarts[pos : pos + 2] in ("Cl", "Br"):
            expr = smarts[pos : pos + 2]
            pos += 2
        else:
            expr = ch
            pos += 1

        idx = len(q_atoms)
        q_atoms.append(expr)
        if prev is not None:
            q_bonds.append((prev, idx, pending_bond))
            pending_bond = ""
        prev = idx

    return q_atoms, q_bonds


def _bond_matches_smarts(mol: MoleculeGraph, u: int, v: int, bpat: str) -> bool:
    bond = mol.get_bond(u, v)
    if bond is None:
        return False
    if bpat == "~":
        return True
    if bpat == "@":
        return bond.in_ring
    if bpat == ":":
        return bond.is_aromatic
    if bpat == "=":
        return (not bond.is_aromatic) and bond.order >= 1.8 and bond.order < 2.5
    if bpat == "#":
        return bond.order >= 2.8
    if bpat == "-":
        return (not bond.is_aromatic) and bond.order < 1.8
    # Unspecified SMARTS bond matches single or aromatic
    return bond.is_aromatic or bond.order < 1.8


def match_smarts(mol: MoleculeGraph, smarts: str, uniquify: bool = True) -> List[Tuple[int, ...]]:
    """Find all substructure matches of a SMARTS pattern in a MoleculeGraph."""
    q_atoms, q_bonds = _parse_smarts_query(smarts)
    if not q_atoms:
        return []

    q_adj: List[List[Tuple[int, str]]] = [[] for _ in q_atoms]
    for u, v, bpat in q_bonds:
        q_adj[u].append((v, bpat))
        q_adj[v].append((u, bpat))

    candidates = [[a.index for a in mol.atoms if _match_atom_expr(mol, a.index, expr)] for expr in q_atoms]
    if any(not c for c in candidates):
        return []

    results: List[Tuple[int, ...]] = []
    seen_sets: Set[frozenset[int]] = set()
    mapping: List[int] = [-1] * len(q_atoms)
    used_atoms: Set[int] = set()

    def _backtrack(q_idx: int) -> None:
        if q_idx == len(q_atoms):
            tup = tuple(mapping)
            if uniquify:
                fset = frozenset(tup)
                if fset in seen_sets:
                    return
                seen_sets.add(fset)
            results.append(tup)
            return

        for m_idx in candidates[q_idx]:
            if m_idx in used_atoms:
                continue
            ok = True
            for q_nbr, bpat in q_adj[q_idx]:
                if q_nbr < q_idx:
                    if not _bond_matches_smarts(mol, m_idx, mapping[q_nbr], bpat):
                        ok = False
                        break
            if not ok:
                continue
            mapping[q_idx] = m_idx
            used_atoms.add(m_idx)
            _backtrack(q_idx + 1)
            used_atoms.remove(m_idx)
            mapping[q_idx] = -1

    _backtrack(0)
    return results


# ---------------------------------------------------------------------------
# 3D Distance-Geometry Embedding & UFF-style Conformer Relaxation
# ---------------------------------------------------------------------------

def identify_rotatable_bonds(mol: MoleculeGraph) -> List[Tuple[int, int]]:
    """Identify acyclic single bonds between non-terminal heavy atoms."""
    rot: List[Tuple[int, int]] = []
    for bond in mol.bonds:
        if bond.in_ring or bond.is_aromatic or bond.order >= 1.5:
            continue
        u, v = bond.atom_u, bond.atom_v
        if len(mol.neighbors(u)) > 1 and len(mol.neighbors(v)) > 1:
            rot.append((u, v))
    return rot


def generate_conformer_3d(mol: MoleculeGraph, seed: int = 42, add_hydrogens: bool = True) -> Conformer3D:
    """Embed a MoleculeGraph into 3D coordinates using spectral distance-geometry
    initialization followed by L-BFGS-B UFF-like bond/angle/steric relaxation.
    """
    rng = np.random.default_rng(seed)
    elements = [a.element for a in mol.atoms]
    charges = np.array([a.partial_charge for a in mol.atoms], dtype=np.float64)
    heavy_indices = list(range(mol.num_atoms))
    bond_tuples: List[Tuple[int, int, float]] = [(b.atom_u, b.atom_v, b.order) for b in mol.bonds]

    if add_hydrogens:
        for atom in mol.atoms:
            for _ in range(atom.implicit_h):
                h_idx = len(elements)
                elements.append("H")
                charges = np.append(charges, 0.05 if atom.element in ("N", "O") else 0.02)
                bond_tuples.append((atom.index, h_idx, 1.0))

    n_tot = len(elements)
    if n_tot == 1:
        return Conformer3D(
            elements=elements,
            coords=np.zeros((1, 3), dtype=np.float64),
            charges=charges,
            heavy_indices=heavy_indices,
            bonds=bond_tuples,
            strain_energy=0.0,
            rotatable_bonds=[],
        )

    # Ideal bond lengths from covalent radii minus bond-order contraction
    ideal_len: Dict[Tuple[int, int], float] = {}
    adj: List[List[int]] = [[] for _ in range(n_tot)]
    for u, v, order in bond_tuples:
        r_u = COVALENT_RADII.get(elements[u], 0.77)
        r_v = COVALENT_RADII.get(elements[v], 0.77)
        bo_corr = 0.0 if order < 1.3 else (0.10 if order < 1.8 else (0.16 if order < 2.5 else 0.25))
        d0 = max(0.95, r_u + r_v - bo_corr)
        ideal_len[(min(u, v), max(u, v))] = d0
        adj[u].append(v)
        adj[v].append(u)

    # Generate 1-3 angle pairs with ideal distances from hybridization
    angle_pairs: List[Tuple[int, int, float]] = []
    for center in range(n_tot):
        nbrs = adj[center]
        if len(nbrs) < 2:
            continue
        if center < mol.num_atoms:
            hyb = mol.atoms[center].hybridization
            theta = math.radians(180.0 if hyb == "sp" else (120.0 if hyb == "sp2" else 109.47))
        else:
            theta = math.radians(109.47)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b = nbrs[i], nbrs[j]
                d_ac = ideal_len[(min(a, center), max(a, center))]
                d_bc = ideal_len[(min(b, center), max(b, center))]
                d_ab = math.sqrt(max(0.5, d_ac ** 2 + d_bc ** 2 - 2.0 * d_ac * d_bc * math.cos(theta)))
                angle_pairs.append((a, b, d_ab))

    # Spectral Laplacian + jitter initialization
    lap = np.zeros((n_tot, n_tot), dtype=np.float64)
    for u, v, _ in bond_tuples:
        lap[u, v] = -1.0
        lap[v, u] = -1.0
        lap[u, u] += 1.0
        lap[v, v] += 1.0
    eigvals, eigvecs = np.linalg.eigh(lap)
    cols = min(3, max(1, n_tot - 1))
    coords0 = np.zeros((n_tot, 3), dtype=np.float64)
    coords0[:, :cols] = eigvecs[:, 1 : 1 + cols] * 2.5
    coords0 += rng.normal(0.0, 0.35, size=(n_tot, 3))

    bonded_set = {(min(u, v), max(u, v)) for u, v, _ in bond_tuples}
    angle_set = {(min(a, b), max(a, b)) for a, b, _ in angle_pairs}
    nonbonded: List[Tuple[int, int, float]] = []
    for i in range(n_tot):
        for j in range(i + 1, n_tot):
            if (i, j) in bonded_set or (i, j) in angle_set:
                continue
            r_min = 0.72 * (COVALENT_RADII.get(elements[i], 0.77) + COVALENT_RADII.get(elements[j], 0.77) + 1.0)
            nonbonded.append((i, j, r_min))

    b_u = np.array([u for u, _, _ in bond_tuples], dtype=np.int64)
    b_v = np.array([v for _, v, _ in bond_tuples], dtype=np.int64)
    b_d0 = np.array([ideal_len[(min(u, v), max(u, v))] for u, v, _ in bond_tuples], dtype=np.float64)

    a_u = np.array([u for u, _, _ in angle_pairs], dtype=np.int64)
    a_v = np.array([v for _, v, _ in angle_pairs], dtype=np.int64)
    a_d0 = np.array([d for _, _, d in angle_pairs], dtype=np.float64)

    nb_u = np.array([u for u, _, _ in nonbonded], dtype=np.int64)
    nb_v = np.array([v for _, v, _ in nonbonded], dtype=np.int64)
    nb_r = np.array([r for _, _, r in nonbonded], dtype=np.float64)

    def _energy_and_grad(flat: FloatArray) -> Tuple[float, FloatArray]:
        x = flat.reshape(n_tot, 3)
        grad = np.zeros_like(x)
        energy = 0.0

        # 1-2 bond stretch
        if len(b_u) > 0:
            diff = x[b_u] - x[b_v]
            dist = np.linalg.norm(diff, axis=1) + 1e-9
            err = dist - b_d0
            energy += float(180.0 * np.sum(err ** 2))
            fac = (360.0 * err / dist)[:, None] * diff
            np.add.at(grad, b_u, fac)
            np.add.at(grad, b_v, -fac)

        # 1-3 angle distance restraint
        if len(a_u) > 0:
            diff = x[a_u] - x[a_v]
            dist = np.linalg.norm(diff, axis=1) + 1e-9
            err = dist - a_d0
            energy += float(45.0 * np.sum(err ** 2))
            fac = (90.0 * err / dist)[:, None] * diff
            np.add.at(grad, a_u, fac)
            np.add.at(grad, a_v, -fac)

        # Non-bonded steric soft repulsion
        if len(nb_u) > 0:
            diff = x[nb_u] - x[nb_v]
            dist = np.linalg.norm(diff, axis=1) + 1e-9
            mask = dist < nb_r
            if np.any(mask):
                viol = nb_r[mask] - dist[mask]
                energy += float(25.0 * np.sum(viol ** 2))
                fac = (-50.0 * viol / dist[mask])[:, None] * diff[mask]
                np.add.at(grad, nb_u[mask], fac)
                np.add.at(grad, nb_v[mask], -fac)

        return energy, grad.ravel()

    res = minimize(_energy_and_grad, coords0.ravel(), method="L-BFGS-B", jac=True, options={"maxiter": 160})
    final_coords = res.x.reshape(n_tot, 3)
    final_coords -= np.mean(final_coords, axis=0, keepdims=True)
    return Conformer3D(
        elements=elements,
        coords=final_coords,
        charges=charges,
        heavy_indices=heavy_indices,
        bonds=bond_tuples,
        strain_energy=float(res.fun),
        rotatable_bonds=identify_rotatable_bonds(mol),
    )


def molecular_weight(mol: MoleculeGraph) -> float:
    """Compute exact monoisotopic/average molecular weight (g/mol)."""
    total = 0.0
    for atom in mol.atoms:
        total += ATOMIC_MASSES.get(atom.element, 12.011) + atom.implicit_h * ATOMIC_MASSES["H"]
    return float(total)


def molecular_formula(mol: MoleculeGraph) -> str:
    """Return Hill-system molecular formula string (e.g., C9H8O4)."""
    counts: Dict[str, int] = {}
    h_count = 0
    for atom in mol.atoms:
        counts[atom.element] = counts.get(atom.element, 0) + 1
        h_count += atom.implicit_h
    if h_count > 0:
        counts["H"] = counts.get("H", 0) + h_count
    parts: List[str] = []
    if "C" in counts:
        c = counts.pop("C")
        parts.append("C" if c == 1 else f"C{c}")
        if "H" in counts:
            h = counts.pop("H")
            parts.append("H" if h == 1 else f"H{h}")
    for elem in sorted(counts):
        cnt = counts[elem]
        parts.append(elem if cnt == 1 else f"{elem}{cnt}")
    return "".join(parts)


def formal_charge_sum(mol: MoleculeGraph) -> int:
    """Return net formal molecular charge."""
    return sum(a.formal_charge for a in mol.atoms)


def electronegativity_matrix(mol: MoleculeGraph) -> FloatArray:
    """Return pairwise Pauling electronegativity difference matrix across heavy atoms."""
    en = np.array([PAULING_EN.get(a.element, 2.5) for a in mol.atoms], dtype=np.float64)
    return np.abs(en[:, None] - en[None, :])
