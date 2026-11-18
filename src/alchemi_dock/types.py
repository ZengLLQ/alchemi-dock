"""Core data structures, chemical constants, and biophysical parameters for AlchemiDock.

All constants use standard IUPAC/SI cheminformatics units:
- Atomic masses: g/mol (Daltons)
- Radii and distances: Angstroms (A)
- Energies: kcal/mol
- Concentrations: micromolar (uM) or nanomolar (nM) as explicitly noted
- Clearance: mL/min/kg or L/h as explicitly noted
- Topological Polar Surface Area (TPSA): A^2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ATOMIC_NUMBERS: Dict[str, int] = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

SYMBOL_BY_Z: Dict[int, str] = {z: sym for sym, z in ATOMIC_NUMBERS.items()}

ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008,
    "B": 10.811,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.065,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.904,
}

VDW_RADII: Dict[str, float] = {
    "H": 1.20,
    "B": 1.92,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}

COVALENT_RADII: Dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}

PAULING_EN: Dict[str, float] = {
    "H": 2.20,
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "Br": 2.96,
    "I": 2.66,
}

DEFAULT_VALENCES: Dict[str, Tuple[int, ...]] = {
    "H": (1,),
    "B": (3,),
    "C": (4,),
    "N": (3, 5),
    "O": (2,),
    "F": (1,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "Cl": (1,),
    "Br": (1,),
    "I": (1,),
}

AROMATIC_ORGANIC_ELEMENTS: Tuple[str, ...] = ("b", "c", "n", "o", "p", "s")
ORGANIC_SUBSET: Tuple[str, ...] = ("B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I")

# Gasteiger-Marsili orbital electronegativity polynomial parameters (a, b, c)
# chi(q) = a + b*q + c*q^2 (eV), damping factor 0.5 per iteration.
GASTEIGER_PARAMS: Dict[Tuple[str, str], Tuple[float, float, float]] = {
    ("H", "sp3"): (7.17, 6.24, -0.56),
    ("C", "sp3"): (7.98, 9.18, 1.88),
    ("C", "sp2"): (8.79, 9.32, 1.51),
    ("C", "sp"): (10.39, 9.45, 0.73),
    ("N", "sp3"): (11.54, 10.82, 1.36),
    ("N", "sp2"): (12.87, 11.15, 0.85),
    ("N", "sp"): (15.68, 11.70, -0.27),
    ("O", "sp3"): (14.18, 12.92, 1.39),
    ("O", "sp2"): (17.07, 13.79, 0.47),
    ("F", "sp3"): (14.66, 13.85, 2.31),
    ("Cl", "sp3"): (11.00, 9.69, 1.35),
    ("Br", "sp3"): (10.08, 8.47, 1.16),
    ("I", "sp3"): (9.90, 7.96, 0.96),
    ("S", "sp3"): (10.14, 9.13, 1.38),
    ("S", "sp2"): (10.88, 9.49, 1.33),
    ("P", "sp3"): (8.90, 8.32, 1.58),
    ("B", "sp2"): (6.50, 7.20, 1.10),
    ("B", "sp3"): (5.98, 6.80, 1.05),
}

# Bickerton et al. (Nature Chemistry 2012) QED asymmetric double-sigmoidal parameters:
# (weight, a, b, c, d, e, f, d_max)
QED_PARAMS: Dict[str, Tuple[float, float, float, float, float, float, float, float]] = {
    "MW": (0.66, 2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561),
    "ALOGP": (0.46, 3.172690585, 137.8624751, 2.534937431, 4.581497897, 0.822739154, 0.576295591, 131.3186604),
    "HBA": (0.05, 2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958, 148.7763046),
    "HBD": (0.61, 1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555, 258.1632616),
    "PSA": (0.06, 1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732, 104.5686167),
    "ROTB": (0.65, 0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403),
    "AROM": (0.48, 3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881, 312.3372610),
    "ALERTS": (0.95, 0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140),
}

# Canonical PAINS / Brenk structural alert SMARTS patterns (name, smarts, severity)
STRUCTURAL_ALERTS: Tuple[Tuple[str, str, str], ...] = (
    ("nitro_aromatic", "[N+](=O)[O-]", "medium"),
    ("michael_acceptor", "C=CC(=O)", "high"),
    ("acyl_halide", "C(=O)[F,Cl,Br,I]", "high"),
    ("aldehyde", "[CH1]=O", "medium"),
    ("peroxide", "OO", "high"),
    ("azide", "N=[N+]=[N-]", "high"),
    ("epoxide_aziridine", "[C;r3]1[O,N;r3][C;r3]1", "high"),
    ("quinone_like", "O=C1C=CC(=O)C=C1", "high"),
    ("hydrazine", "[N;X3][N;X3]", "medium"),
    ("thiocarbonyl", "C=S", "low"),
)


@dataclass
class AtomRecord:
    """Single atom node in a molecular graph."""

    index: int
    element: str
    formal_charge: int = 0
    is_aromatic: bool = False
    explicit_h: Optional[int] = None
    implicit_h: int = 0
    isotope: Optional[int] = None
    chirality: Optional[str] = None
    hybridization: str = "sp3"
    in_ring: bool = False
    ring_sizes: Tuple[int, ...] = ()
    partial_charge: float = 0.0


@dataclass
class BondRecord:
    """Undirected covalent bond edge in a molecular graph."""

    atom_u: int
    atom_v: int
    order: float = 1.0
    is_aromatic: bool = False
    is_conjugated: bool = False
    in_ring: bool = False
    stereo: Optional[str] = None


@dataclass
class MoleculeGraph:
    """Pure-Python graph representation of a chemical structure."""

    smiles: str
    atoms: List[AtomRecord]
    bonds: List[BondRecord]
    sssr_rings: List[Tuple[int, ...]] = field(default_factory=list)
    canonical_smiles: str = ""
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_atoms(self) -> int:
        """Return heavy atom count in the parsed molecular graph."""
        return len(self.atoms)

    @property
    def num_bonds(self) -> int:
        """Return covalent bond count in the parsed molecular graph."""
        return len(self.bonds)

    def neighbors(self, atom_idx: int) -> List[int]:
        """Return neighbor atom indices adjacent to atom_idx."""
        nbrs: List[int] = []
        for bond in self.bonds:
            if bond.atom_u == atom_idx:
                nbrs.append(bond.atom_v)
            elif bond.atom_v == atom_idx:
                nbrs.append(bond.atom_u)
        return nbrs

    def neighbor_bonds(self, atom_idx: int) -> List[Tuple[int, BondRecord]]:
        """Return (neighbor_idx, bond) pairs for atom_idx."""
        out: List[Tuple[int, BondRecord]] = []
        for bond in self.bonds:
            if bond.atom_u == atom_idx:
                out.append((bond.atom_v, bond))
            elif bond.atom_v == atom_idx:
                out.append((bond.atom_u, bond))
        return out

    def get_bond(self, u: int, v: int) -> Optional[BondRecord]:
        """Return the BondRecord between atoms u and v, or None."""
        for bond in self.bonds:
            if (bond.atom_u == u and bond.atom_v == v) or (bond.atom_u == v and bond.atom_v == u):
                return bond
        return None


@dataclass
class Conformer3D:
    """3D molecular conformer including implicit/explicit hydrogens."""

    elements: List[str]
    coords: FloatArray
    charges: FloatArray
    heavy_indices: List[int]
    bonds: List[Tuple[int, int, float]]
    strain_energy: float
    rotatable_bonds: List[Tuple[int, int]]


@dataclass
class FingerprintSet:
    """Circular Morgan ECFP and structural key fingerprints."""

    ecfp4_bits: IntArray
    ecfp4_counts: Dict[int, int]
    ecfp6_bits: IntArray
    fcfp4_bits: IntArray
    maccs_bits: IntArray
    n_bits: int = 1024


@dataclass
class ScaffoldDecomposition:
    """Bemis-Murcko scaffold and matched molecular fragmentation."""

    original_smiles: str
    murcko_smiles: str
    generic_skeleton_smiles: str
    ring_system_count: int
    linker_atom_count: int
    sidechain_atom_count: int


@dataclass
class MMPTransform:
    """Single Matched Molecular Pair transformation rule."""

    core_smiles: str
    from_fragment: str
    to_fragment: str
    from_smiles: str
    to_smiles: str
    delta_property: float
    property_name: str


@dataclass
class GNNMetrics:
    """Evaluation metrics for QSAR regression/classification models."""

    rmse: float
    mae: float
    r2: float
    spearman_rho: float
    auroc: float
    enrichment_5pct: float


@dataclass
class GNNTrainingReport:
    """Training and scaffold-split validation report for a GNN model."""

    architecture: str
    epochs: int
    train_losses: List[float]
    val_losses: List[float]
    train_metrics: GNNMetrics
    val_metrics: GNNMetrics
    atom_attributions: Dict[str, List[float]]


@dataclass
class ADMEProfile:
    """Multitask ADME, PK, and safety liability prediction."""

    smiles: str
    mw: float
    clogp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    fsp3: float
    qed: float
    sa_score: float
    ro5_violations: int
    alerts: List[str]
    log_s_esol: float
    papp_caco2_1e6_cms: float
    log_bb: float
    cns_mpo: float
    fu_plasma: float
    cl_int_ul_min_mg: float
    hepatic_extraction_eh: float
    oral_bioavailability_f: float
    herg_risk_prob: float
    cyp3a4_inhibition_prob: float
    cyp_ddi_auc_ratio: float
    ionization_class: str
    estimated_pka: float


@dataclass
class PKCurveResult:
    """Simulated multi-dose 1-compartment oral pharmacokinetic trajectory."""

    times_h: FloatArray
    plasma_conc_uM: FloatArray
    cmax_uM: float
    tmax_h: float
    auc_0_inf_uM_h: float
    half_life_h: float
    clearance_L_h: float
    therapeutic_index_margin: float


@dataclass
class PocketAtom:
    """Receptor pocket atom with empirical interaction typing."""

    index: int
    residue_name: str
    atom_name: str
    element: str
    coord: FloatArray
    partial_charge: float
    is_hbd: bool = False
    is_hba: bool = False
    is_hydrophobic: bool = False
    is_aromatic: bool = False


@dataclass
class BindingPocket:
    """Target protein binding pocket geometry and grid bounds."""

    name: str
    target_family: str
    center: FloatArray
    box_size: FloatArray
    atoms: List[PocketAtom]


@dataclass
class InteractionContact:
    """Receptor-ligand non-covalent interaction record."""

    interaction_type: str
    ligand_atom_idx: int
    ligand_element: str
    residue_name: str
    pocket_atom_name: str
    distance_a: float
    energy_kcal_mol: float


@dataclass
class DockingPose:
    """Empirical Vina-like docked pose and interaction breakdown."""

    smiles: str
    pocket_name: str
    total_score_kcal_mol: float
    gauss1_term: float
    gauss2_term: float
    repulsion_term: float
    hydrophobic_term: float
    hbond_term: float
    electrostatic_term: float
    torsion_penalty: float
    ligand_coords: FloatArray
    contacts: List[InteractionContact]
    rmsd_to_reference: Optional[float] = None


@dataclass
class ProtacSpec:
    """Biophysical specification of a heterobifunctional PROTAC degrader."""

    name: str
    warhead_smiles: str
    linker_smiles: str
    e3_ligand_smiles: str
    kd_poi_uM: float
    kd_e3_uM: float
    poi_total_uM: float = 0.5
    e3_total_uM: float = 0.2
    cooperativity_alpha: Optional[float] = None
    k_ub_per_h: float = 1.8
    k_syn_uM_per_h: float = 0.15
    k_deg_basal_per_h: float = 0.30


@dataclass
class ProtacEquilibriumCurve:
    """Douglass-Miller ternary complex equilibrium dose-response curve."""

    spec: ProtacSpec
    estimated_alpha: float
    conc_grid_uM: FloatArray
    ternary_ple_uM: FloatArray
    binary_pl_uM: FloatArray
    binary_el_uM: FloatArray
    peak_ternary_uM: float
    hook_conc_uM: float
    peak_fraction_e3_bound: float
    mass_balance_max_err: float


@dataclass
class ProtacDegradationResult:
    """Time-course POI proteasomal degradation kinetics across PROTAC doses."""

    curve: ProtacEquilibriumCurve
    dose_uM: float
    times_h: FloatArray
    poi_fraction_remaining: FloatArray
    dose_grid_uM: FloatArray
    dmax_fraction: float
    dc50_uM: float
    degradation_aoc: float


@dataclass
class OptimizationCandidate:
    """Single evaluated candidate along a de novo optimization trajectory."""

    smiles: str
    generation: int
    parent_smiles: str
    edit_action: str
    predicted_pic50: float
    qed: float
    sa_score: float
    clogp: float
    tpsa: float
    adme_penalty: float
    mpo_score: float
    scaffold_smiles: str
    pareto_rank: int = 0


@dataclass
class OptimizationTrajectory:
    """Complete MCTS + evolutionary Pareto optimization history."""

    seed_smiles: str
    algorithm: str
    generations: int
    candidates_evaluated: int
    valid_rate: float
    mean_diversity: float
    best_candidate: OptimizationCandidate
    pareto_frontier: List[OptimizationCandidate]
    generation_best_mpo: List[float]
    generation_mean_mpo: List[float]


@dataclass
class ConformalInterval:
    """Distribution-free split conformal prediction interval."""

    prediction: float
    lower: float
    upper: float
    alpha: float
    quantile_radius: float


@dataclass
class SensitivityReport:
    """Saltelli Sobol first-order and total-order sensitivity indices."""

    parameter_names: List[str]
    first_order_si: Dict[str, float]
    total_order_sti: Dict[str, float]
    output_mean: float
    output_std: float


@dataclass
class CampaignReport:
    """Top-level multi-stage computational drug discovery campaign report."""

    seed: int
    library_size: int
    scaffold_count: int
    cluster_count: int
    gnn_report: GNNTrainingReport
    conformal_coverage: float
    lead_adme: ADMEProfile
    lead_pk: PKCurveResult
    lead_docking: DockingPose
    protac_degradation: ProtacDegradationResult
    optimization: OptimizationTrajectory
    sensitivity: SensitivityReport
