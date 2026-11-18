"""AlchemiDock: Pure-Python computational drug discovery and cheminformatics platform."""

from .adme import predict_adme_profile, simulate_oral_pk
from .benchmark import (
    compute_planted_sar_pic50,
    default_protac_benchmark_specs,
    load_kinase_benchmark_dataset,
)
from .calibration import (
    calibrate_parameters_nelder_mead,
    conformal_predict_intervals,
    empirical_conformal_coverage,
    saltelli_sobol_sensitivity,
    split_conformal_quantile,
)
from .denovo import optimize_molecules_mcts_pareto, propose_valid_mutations
from .descriptors import (
    bemis_murcko_scaffold,
    butina_clustering,
    compute_clogp_and_mr,
    compute_fingerprint_set,
    compute_morgan_fingerprint,
    compute_qed,
    compute_sa_score,
    compute_tpsa,
    mine_matched_molecular_pairs,
    tanimoto_similarity,
)
from .docking import build_kinase_atp_pocket, dock_ligand, kabsch_rmsd, score_ligand_pose
from .gnn import MolecularGNN, featurize_mol_tensor, has_torch_adapter, scaffold_split_indices
from .molgraph import (
    assign_kekule_bonds,
    canonical_smiles,
    compute_gasteiger_charges,
    compute_sssr,
    generate_conformer_3d,
    match_smarts,
    parse_smiles,
    perceive_aromaticity,
)
from .pipeline import campaign_report_to_dict, run_full_discovery_campaign
from .protac import (
    compute_protac_equilibrium_curve,
    estimate_protac_cooperativity,
    simulate_protac_degradation,
    solve_ternary_equilibrium_point,
)
from .visualizer import (
    assert_ascii,
    render_campaign_dashboard,
    render_docking_profile,
    render_molecule_summary,
    render_pareto_scatter,
    render_pk_curve,
    render_protac_hook_curve,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "MolecularGNN",
    "assign_kekule_bonds",
    "assert_ascii",
    "bemis_murcko_scaffold",
    "build_kinase_atp_pocket",
    "butina_clustering",
    "calibrate_parameters_nelder_mead",
    "campaign_report_to_dict",
    "canonical_smiles",
    "compute_clogp_and_mr",
    "compute_fingerprint_set",
    "compute_gasteiger_charges",
    "compute_morgan_fingerprint",
    "compute_planted_sar_pic50",
    "compute_protac_equilibrium_curve",
    "compute_qed",
    "compute_sa_score",
    "compute_sssr",
    "compute_tpsa",
    "conformal_predict_intervals",
    "default_protac_benchmark_specs",
    "dock_ligand",
    "empirical_conformal_coverage",
    "estimate_protac_cooperativity",
    "featurize_mol_tensor",
    "generate_conformer_3d",
    "has_torch_adapter",
    "kabsch_rmsd",
    "load_kinase_benchmark_dataset",
    "match_smarts",
    "mine_matched_molecular_pairs",
    "optimize_molecules_mcts_pareto",
    "parse_smiles",
    "perceive_aromaticity",
    "predict_adme_profile",
    "propose_valid_mutations",
    "render_campaign_dashboard",
    "render_docking_profile",
    "render_molecule_summary",
    "render_pareto_scatter",
    "render_pk_curve",
    "render_protac_hook_curve",
    "run_full_discovery_campaign",
    "saltelli_sobol_sensitivity",
    "scaffold_split_indices",
    "score_ligand_pose",
    "simulate_oral_pk",
    "simulate_protac_degradation",
    "solve_ternary_equilibrium_point",
    "split_conformal_quantile",
    "tanimoto_similarity",
]
