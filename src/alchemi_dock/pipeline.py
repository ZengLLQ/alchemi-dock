"""End-to-end computational drug discovery campaign orchestrator.

Coordinates molecular graph perception, scaffold/Butina diversity clustering,
MMP SAR mining, GNN training with split conformal prediction intervals, ADME/PK
simulation, structure-based docking, PROTAC ternary degradation modeling, MCTS/Pareto
de novo optimization, and Saltelli Sobol sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import numpy as np

from .adme import predict_adme_profile, simulate_oral_pk
from .benchmark import (
    compute_planted_sar_pic50,
    default_protac_benchmark_specs,
    load_kinase_benchmark_dataset,
)
from .calibration import (
    conformal_predict_intervals,
    empirical_conformal_coverage,
    saltelli_sobol_sensitivity,
    split_conformal_quantile,
)
from .denovo import optimize_molecules_mcts_pareto
from .descriptors import (
    bemis_murcko_scaffold,
    butina_clustering,
    compute_fingerprint_set,
    mine_matched_molecular_pairs,
)
from .docking import build_kinase_atp_pocket, dock_ligand
from .gnn import MolecularGNN, featurize_mol_tensor, scaffold_split_indices
from .molgraph import parse_smiles
from .protac import solve_ternary_equilibrium_point, simulate_protac_degradation
from .types import CampaignReport, FloatArray


def run_full_discovery_campaign(
    architecture: str = "gatv2",
    epochs: int = 24,
    opt_generations: int = 6,
    seed: int = 42,
) -> CampaignReport:
    """Execute a complete reproducible multi-stage AI drug discovery campaign."""
    _, smiles_list, targets = load_kinase_benchmark_dataset(seed=seed)

    # 1. Graph perception, Fingerprints, Murcko Scaffolds, Butina Clustering & MMP
    mols = [parse_smiles(smi) for smi in smiles_list]
    fps = [compute_fingerprint_set(m).ecfp4_bits for m in mols]
    scaffolds = {bemis_murcko_scaffold(m).murcko_smiles for m in mols}
    clusters = butina_clustering(fps, similarity_cutoff=0.45)
    _ = mine_matched_molecular_pairs(smiles_list, targets.tolist(), property_name="pIC50")

    # 2. Scaffold-split GNN SAR training + Split Conformal calibration
    graphs = [featurize_mol_tensor(m) for m in mols]
    train_idx, val_idx = scaffold_split_indices(smiles_list, val_fraction=0.28)
    train_graphs = [graphs[i] for i in train_idx]
    train_y = targets[train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    val_y = targets[val_idx]

    gnn = MolecularGNN(architecture=architecture, hidden_dim=28, num_layers=2, num_heads=2, seed=seed)
    gnn_report = gnn.fit(train_graphs, train_y, val_graphs, val_y, epochs=epochs, lr=9e-3)

    # Split conformal calibration using half val as calibration, half as held-out test
    val_preds, _ = gnn.predict(val_graphs)
    mid = max(1, len(val_preds) // 2)
    q_rad = split_conformal_quantile(val_y[:mid], val_preds[:mid], alpha=0.15)
    intervals = conformal_predict_intervals(val_preds[mid:], q_rad, alpha=0.15)
    coverage = empirical_conformal_coverage(val_y[mid:], intervals) if len(val_y[mid:]) > 0 else 1.0

    # 3. De novo MCTS + Pareto optimization from quinazoline hinge seed
    seed_smi = "Fc1ccc(Nc2ncnc3ccccc23)cc1"

    def _gnn_surrogate(smi: str) -> float:
        g = featurize_mol_tensor(parse_smiles(smi))
        pred, _, _ = gnn.forward_single(g)
        # Blend GNN prediction with biophysical SAR anchor
        return 0.45 * pred + 0.55 * compute_planted_sar_pic50(smi)

    opt_traj = optimize_molecules_mcts_pareto(
        seed_smiles=seed_smi,
        generations=opt_generations,
        population_size=14,
        potency_fn=_gnn_surrogate,
        seed=seed,
    )

    # 4. Lead ADME, Oral PK ODE & Pocket Docking
    lead_smi = opt_traj.best_candidate.smiles
    lead_adme = predict_adme_profile(lead_smi)
    lead_pk = simulate_oral_pk(lead_adme, dose_mg=100.0, dosing_interval_h=12.0, n_doses=3)
    pocket = build_kinase_atp_pocket()
    lead_docking = dock_ligand(lead_smi, pocket=pocket, mc_steps=110, seed=seed)

    # 5. PROTAC Ternary Cooperativity & Degradation Kinetics
    protac_spec = default_protac_benchmark_specs()["optimal_rigid_linker"]
    protac_deg = simulate_protac_degradation(protac_spec, hours=24.0)

    # 6. Saltelli Sobol Global Sensitivity over PROTAC Ternary Peak Complex
    def _ternary_peak_fn(theta: FloatArray) -> float:
        kd_p, kd_e, alpha, l0 = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])
        _, _, _, _, _, ple = solve_ternary_equilibrium_point(l0, 0.5, 0.2, kd_p, kd_e, alpha)
        return ple

    sens = saltelli_sobol_sensitivity(
        _ternary_peak_fn,
        param_names=["kd_poi_uM", "kd_e3_uM", "alpha_coop", "dose_uM"],
        bounds=[(0.01, 0.25), (0.02, 0.40), (0.5, 15.0), (0.05, 1.50)],
        n_samples=32,
        seed=seed,
    )

    return CampaignReport(
        seed=seed,
        library_size=len(smiles_list),
        scaffold_count=len(scaffolds),
        cluster_count=len(clusters),
        gnn_report=gnn_report,
        conformal_coverage=coverage,
        lead_adme=lead_adme,
        lead_pk=lead_pk,
        lead_docking=lead_docking,
        protac_degradation=protac_deg,
        optimization=opt_traj,
        sensitivity=sens,
    )


def campaign_report_to_dict(report: CampaignReport) -> Dict[str, Any]:
    """Serialize a CampaignReport into a JSON-safe dictionary."""
    raw = asdict(report)

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    sanitized: Dict[str, Any] = _sanitize(raw)
    return sanitized
