"""Pure 7-bit ASCII terminal visualizer for AlchemiDock scientific outputs.

Strictly uses ASCII codes 0-127 (+, -, |, *, #, ., :, =, ^, v) to ensure 100%
compatibility across headless CI runners and Windows cp1252 consoles.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np

from .molgraph import molecular_formula, molecular_weight
from .types import (
    ADMEProfile,
    CampaignReport,
    DockingPose,
    FloatArray,
    MoleculeGraph,
    OptimizationTrajectory,
    PKCurveResult,
    ProtacDegradationResult,
    SensitivityReport,
)


def assert_ascii(text: str) -> str:
    """Validate that text contains strictly 7-bit ASCII characters (0-127)."""
    for idx, ch in enumerate(text):
        if ord(ch) > 127:
            raise ValueError(f"Non-ASCII character {repr(ch)} (ord={ord(ch)}) at index {idx}")
    return text


def _ascii_bar(frac: float, width: int = 20) -> str:
    clamped = max(0.0, min(1.0, frac))
    filled = int(round(clamped * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _ascii_xy_plot(
    x_vals: Union[Sequence[float], FloatArray],
    y_vals: Union[Sequence[float], FloatArray],
    width: int = 58,
    height: int = 14,
    title: str = "",
    x_label: str = "x",
    y_label: str = "y",
    marker: str = "*",
) -> List[str]:
    """Render a crisp 7-bit ASCII 2D line/point plot."""
    x_arr = np.asarray(x_vals, dtype=np.float64)
    y_arr = np.asarray(y_vals, dtype=np.float64)
    x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
    y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))
    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for x, y in zip(x_arr, y_arr):
        col = int(round((x - x_min) / (x_max - x_min) * (width - 1)))
        row = height - 1 - int(round((y - y_min) / (y_max - y_min) * (height - 1)))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = marker

    lines: List[str] = []
    if title:
        lines.append(f"  {title}")
    for r in range(height):
        val = y_max - (r / max(1, height - 1)) * (y_max - y_min)
        prefix = f"{val:7.3f} |" if r in (0, height // 2, height - 1) else "        |"
        lines.append(prefix + "".join(grid[r]) + "|")
    lines.append("        +" + ("-" * width) + "+")
    lines.append(f"         {x_min:<10.3f}  [{x_label} vs {y_label}]  {x_max:>10.3f}")
    return lines


def render_molecule_summary(
    mol: MoleculeGraph,
    adme: Optional[ADMEProfile] = None,
    attributions: Optional[Sequence[float]] = None,
) -> str:
    """Render a 7-bit ASCII molecular graph topology, Gasteiger charge, and ADME table."""
    lines = [
        "=" * 76,
        f"MOLECULAR GRAPH & PHYSICOCHEMICAL PROFILE: {mol.canonical_smiles or mol.smiles}",
        "=" * 76,
        f"  Formula       : {molecular_formula(mol)}   MW = {molecular_weight(mol):.2f} g/mol",
        f"  Topology      : {mol.num_atoms} heavy atoms | {mol.num_bonds} bonds | {len(mol.sssr_rings)} SSSR rings",
        f"  Aromatic Atoms: {sum(1 for a in mol.atoms if a.is_aromatic)} | "
        f"Charge Sum = {sum(a.partial_charge for a in mol.atoms):+.6f}",
    ]
    if adme is not None:
        lines.extend(
            [
                f"  Descriptors   : cLogP={adme.clogp:.2f} | TPSA={adme.tpsa:.1f} A^2 | "
                f"HBD={adme.hbd} | HBA={adme.hba} | RotB={adme.rotatable_bonds} | Fsp3={adme.fsp3:.2f}",
                f"  Drug-Likeness : QED={adme.qed:.3f} {_ascii_bar(adme.qed, 14)} | "
                f"SAscore={adme.sa_score:.2f}/10 | Ro5_Viol={adme.ro5_violations}",
                f"  ADME / Safety : logS={adme.log_s_esol:.2f} | Papp={adme.papp_caco2_1e6_cms:.1f}e-6 cm/s | "
                f"logBB={adme.log_bb:.2f} | CNS_MPO={adme.cns_mpo:.2f}/6",
                f"  PK / Tox      : fu={adme.fu_plasma:.3f} | CLint={adme.cl_int_ul_min_mg:.1f} uL/min/mg | "
                f"F_oral={adme.oral_bioavailability_f:.2f} | hERG={adme.herg_risk_prob:.2f} | "
                f"CYP3A4={adme.cyp3a4_inhibition_prob:.2f}",
                f"  Ionization    : {adme.ionization_class} (est pKa={adme.estimated_pka:.1f}) | "
                f"Alerts: {', '.join(adme.alerts) if adme.alerts else 'None'}",
            ]
        )

    lines.append("-" * 76)
    lines.append("  Idx  Elem  Hyb  Arom  Ring  H_imp  Gasteiger_q   Attribution")
    max_attr = max(attributions) if attributions and max(attributions) > 0 else 1.0
    for i, atom in enumerate(mol.atoms[:18]):
        attr_str = ""
        if attributions is not None and i < len(attributions):
            attr_str = f"{attributions[i]:.3f} " + _ascii_bar(attributions[i] / max_attr, 12)
        lines.append(
            f"  {i:3d}  {atom.element:<4s}  {atom.hybridization:<3s}  "
            f"{'Y' if atom.is_aromatic else 'N':<4s}  {'Y' if atom.in_ring else 'N':<4s}  "
            f"{atom.implicit_h:5d}  {atom.partial_charge:+11.4f}   {attr_str}"
        )
    if mol.num_atoms > 18:
        lines.append(f"  ... ({mol.num_atoms - 18} additional heavy atoms omitted)")
    return assert_ascii("\n".join(lines))


def render_pk_curve(pk: PKCurveResult) -> str:
    """Render 7-bit ASCII oral multi-dose plasma concentration-time profile."""
    lines = [
        "=" * 76,
        "1-COMPARTMENT ORAL PHARMACOKINETIC TRAJECTORY",
        "=" * 76,
        f"  Cmax={pk.cmax_uM:.3f} uM @ Tmax={pk.tmax_h:.1f} h | AUC={pk.auc_0_inf_uM_h:.2f} uM*h | "
        f"t1/2={pk.half_life_h:.2f} h | CL_H={pk.clearance_L_h:.1f} L/h | Margin={pk.therapeutic_index_margin:.1f}x",
    ]
    lines.extend(
        _ascii_xy_plot(
            pk.times_h,
            pk.plasma_conc_uM,
            title="Plasma Concentration [uM] vs Time [h]",
            x_label="Time (h)",
            y_label="Conc (uM)",
            marker="+",
        )
    )
    return assert_ascii("\n".join(lines))


def render_docking_profile(pose: DockingPose) -> str:
    """Render 7-bit ASCII Vina empirical score breakdown and PLIF contacts."""
    lines = [
        "=" * 76,
        f"STRUCTURE-BASED EMPIRICAL DOCKING: {pose.pocket_name}",
        "=" * 76,
        f"  SMILES      : {pose.smiles}",
        f"  Total Score : {pose.total_score_kcal_mol:+.3f} kcal/mol",
        f"  Terms       : Gauss1={pose.gauss1_term:+.3f} | Gauss2={pose.gauss2_term:+.3f} | "
        f"Repul={pose.repulsion_term:+.3f} | Hydrophob={pose.hydrophobic_term:+.3f}",
        f"                HBond={pose.hbond_term:+.3f} | Elec={pose.electrostatic_term:+.3f} | "
        f"TorsPen={pose.torsion_penalty:.3f}",
        "-" * 76,
        "  Type          LigAtom  PocketResidue   Dist(A)   E_contrib(kcal/mol)",
    ]
    for c in pose.contacts[:10]:
        lines.append(
            f"  {c.interaction_type:<13s} {c.ligand_element}{c.ligand_atom_idx:<6d} "
            f"{c.residue_name}:{c.pocket_atom_name:<8s} {c.distance_a:6.2f}    {c.energy_kcal_mol:+.3f}"
        )
    if not pose.contacts:
        lines.append("  (No short-range contacts < 4.5 A recorded)")
    return assert_ascii("\n".join(lines))


def render_protac_hook_curve(deg: ProtacDegradationResult) -> str:
    """Render 7-bit ASCII PROTAC ternary Hook-effect curve and POI degradation profile."""
    c = deg.curve
    log_conc = [math.log10(max(1e-6, float(x))) for x in c.conc_grid_uM]
    lines = [
        "=" * 76,
        f"PROTAC TERNARY COMPLEX & DEGRADATION KINETICS: {c.spec.name}",
        "=" * 76,
        f"  Cooperativity alpha : {c.estimated_alpha:.2f} | Hook Peak = {c.peak_ternary_uM:.4f} uM "
        f"@ [PROTAC]={c.hook_conc_uM:.4f} uM",
        f"  Mass-Balance Err    : {c.mass_balance_max_err:.2e} uM | "
        f"E3 Bound Peak = {100.0 * c.peak_fraction_e3_bound:.1f}%",
        f"  Degradation Metrics : Dmax = {100.0*deg.dmax_fraction:.1f}% | DC50 = {deg.dc50_uM:.4f} uM | "
        f"AOC = {deg.degradation_aoc:.3f}",
    ]
    lines.extend(
        _ascii_xy_plot(
            log_conc,
            c.ternary_ple_uM,
            title="Ternary Complex [POI-PROTAC-E3] Hook Curve vs log10([PROTAC] uM)",
            x_label="log10(uM)",
            y_label="[PLE] uM",
            marker="*",
        )
    )
    lines.extend(
        _ascii_xy_plot(
            deg.times_h,
            deg.poi_fraction_remaining * 100.0,
            title=f"Target POI Remaining (%) Over 24h @ Dose={deg.dose_uM:.4f} uM",
            x_label="Time (h)",
            y_label="POI (%)",
            marker="#",
        )
    )
    return assert_ascii("\n".join(lines))


def render_pareto_scatter(traj: OptimizationTrajectory) -> str:
    """Render 7-bit ASCII 2D Pareto frontier scatter (pIC50 vs QED) and trajectory."""
    lines = [
        "=" * 76,
        f"DE NOVO MCTS + PARETO OPTIMIZATION ({traj.algorithm})",
        "=" * 76,
        f"  Seed SMILES : {traj.seed_smiles}",
        f"  Best SMILES : {traj.best_candidate.smiles}",
        f"  Best Metrics: pIC50={traj.best_candidate.predicted_pic50:.2f} | QED={traj.best_candidate.qed:.3f} | "
        f"SA={traj.best_candidate.sa_score:.2f} | MPO={traj.best_candidate.mpo_score:.3f}",
        f"  Search Stats: Evaluated={traj.candidates_evaluated} | ValidRate={100.0*traj.valid_rate:.1f}% | "
        f"Diversity={traj.mean_diversity:.3f} | ParetoFront={len(traj.pareto_frontier)}",
    ]

    pts = traj.pareto_frontier
    if pts:
        width, height = 56, 12
        xs = [c.predicted_pic50 for c in pts]
        ys = [c.qed for c in pts]
        x_min, x_max = min(xs) - 0.1, max(xs) + 0.1
        y_min, y_max = max(0.0, min(ys) - 0.05), min(1.0, max(ys) + 0.05)
        grid = [[" " for _ in range(width)] for _ in range(height)]
        for c in pts:
            col = int(round((c.predicted_pic50 - x_min) / max(1e-9, x_max - x_min) * (width - 1)))
            row = height - 1 - int(round((c.qed - y_min) / max(1e-9, y_max - y_min) * (height - 1)))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            grid[row][col] = "*" if c.smiles != traj.best_candidate.smiles else "B"

        lines.append("  Pareto Frontier: QED (y) vs Predicted pIC50 (x)  [B=Best, *=Pareto]")
        for r in range(height):
            val = y_max - (r / max(1, height - 1)) * (y_max - y_min)
            lbl = f"{val:6.2f} |" if r in (0, height // 2, height - 1) else "       |"
            lines.append(lbl + "".join(grid[r]) + "|")
        lines.append("       +" + ("-" * width) + "+")
        lines.append(f"        {x_min:<8.2f}   [pIC50]                        {x_max:>8.2f}")

    lines.append("  MPO Progression by Gen: " + " -> ".join(f"{v:.3f}" for v in traj.generation_best_mpo))
    return assert_ascii("\n".join(lines))


def render_sensitivity_table(sens: SensitivityReport) -> str:
    """Render 7-bit ASCII Saltelli Sobol global sensitivity indices."""
    lines = [
        "=" * 76,
        f"SALTELLI SOBOL SENSITIVITY (mean={sens.output_mean:.4f}, std={sens.output_std:.4f})",
        "=" * 76,
        "  Parameter         S_i (First-Order)   S_Ti (Total-Order)  Bar(S_Ti)",
    ]
    for name in sens.parameter_names:
        si = sens.first_order_si[name]
        sti = sens.total_order_sti[name]
        lines.append(f"  {name:<16s}  {si:16.3f}    {sti:16.3f}    {_ascii_bar(sti, 16)}")
    return assert_ascii("\n".join(lines))


def render_campaign_dashboard(report: CampaignReport) -> str:
    """Render an integrated 7-bit ASCII executive dashboard for a CampaignReport."""
    g = report.gnn_report
    lines = [
        "#" * 76,
        "ALCHEMIDOCK - COMPUTATIONAL DRUG DISCOVERY EXECUTIVE DASHBOARD",
        "#" * 76,
        f"  Library       : {report.library_size} compounds | {report.scaffold_count} Murcko scaffolds | "
        f"{report.cluster_count} Butina clusters",
        f"  GNN ({g.architecture.upper():<5s})   : Train RMSE={g.train_metrics.rmse:.3f} "
        f"(R2={g.train_metrics.r2:.3f}) | "
        f"Val RMSE={g.val_metrics.rmse:.3f} (Spearman={g.val_metrics.spearman_rho:.3f}, "
        f"AUROC={g.val_metrics.auroc:.3f})",
        f"  Conformal UQ  : Realized Held-Out Coverage = {100.0 * report.conformal_coverage:.1f}%",
        f"  Lead Compound : {report.lead_adme.smiles}",
        f"  Lead ADME/PK  : QED={report.lead_adme.qed:.3f} | cLogP={report.lead_adme.clogp:.2f} | "
        f"F_oral={report.lead_adme.oral_bioavailability_f:.2f} | Cmax={report.lead_pk.cmax_uM:.2f} uM | "
        f"hERG={report.lead_adme.herg_risk_prob:.2f}",
        f"  Docking Pose  : {report.lead_docking.total_score_kcal_mol:+.2f} kcal/mol "
        f"({len(report.lead_docking.contacts)} contacts in {report.lead_docking.pocket_name})",
        f"  PROTAC Twin   : alpha={report.protac_degradation.curve.estimated_alpha:.2f} | "
        f"Dmax={100.0*report.protac_degradation.dmax_fraction:.1f}% | "
        f"DC50={report.protac_degradation.dc50_uM:.4f} uM",
        "",
        render_pareto_scatter(report.optimization),
        "",
        render_protac_hook_curve(report.protac_degradation),
        "",
        render_sensitivity_table(report.sensitivity),
    ]
    return assert_ascii("\n".join(lines))
