"""Command-line interface for AlchemiDock (`alchemi-dock` and `python -m alchemi_dock.cli`).

Provides 7-bit ASCII terminal outputs and optional JSON telemetry export across
7 operational subcommands: profile, screen, dock, protac, optimize, benchmark, demo.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from .adme import predict_adme_profile, simulate_oral_pk
from .benchmark import default_protac_benchmark_specs, load_kinase_benchmark_dataset
from .calibration import (
    conformal_predict_intervals,
    empirical_conformal_coverage,
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
from .pipeline import campaign_report_to_dict, run_full_discovery_campaign
from .protac import simulate_protac_degradation
from .types import ProtacSpec
from .visualizer import (
    render_campaign_dashboard,
    render_docking_profile,
    render_molecule_summary,
    render_pareto_scatter,
    render_pk_curve,
    render_protac_hook_curve,
)


def _cmd_profile(args: argparse.Namespace) -> int:
    mol = parse_smiles(args.smiles)
    adme = predict_adme_profile(args.smiles)
    pk = simulate_oral_pk(adme, dose_mg=args.dose_mg)
    print(render_molecule_summary(mol, adme=adme))
    print("")
    print(render_pk_curve(pk))
    if args.json:
        payload = {"smiles": adme.smiles, "qed": adme.qed, "clogp": adme.clogp, "cmax_uM": pk.cmax_uM}
        print(json.dumps(payload, indent=2))
    return 0


def _cmd_screen(args: argparse.Namespace) -> int:
    _, smiles_list, targets = load_kinase_benchmark_dataset(seed=args.seed)
    mols = [parse_smiles(s) for s in smiles_list]
    fps = [compute_fingerprint_set(m).ecfp4_bits for m in mols]
    clusters = butina_clustering(fps, similarity_cutoff=args.cutoff)
    scaffolds = {bemis_murcko_scaffold(m).murcko_smiles for m in mols}
    mmps = mine_matched_molecular_pairs(smiles_list, targets.tolist(), property_name="pIC50")

    graphs = [featurize_mol_tensor(m) for m in mols]
    tr_idx, va_idx = scaffold_split_indices(smiles_list, val_fraction=0.28)
    gnn = MolecularGNN(architecture=args.arch, hidden_dim=28, num_layers=2, num_heads=2, seed=args.seed)
    report = gnn.fit(
        [graphs[i] for i in tr_idx],
        targets[tr_idx],
        [graphs[i] for i in va_idx],
        targets[va_idx],
        epochs=args.epochs,
        lr=9e-3,
    )
    va_preds, _ = gnn.predict([graphs[i] for i in va_idx])
    mid = max(1, len(va_preds) // 2)
    q_rad = split_conformal_quantile(targets[va_idx][:mid], va_preds[:mid], alpha=0.15)
    intervals = conformal_predict_intervals(va_preds[mid:], q_rad, alpha=0.15)
    cov = empirical_conformal_coverage(targets[va_idx][mid:], intervals) if len(va_idx) > mid else 1.0

    print(f"Library: {len(mols)} molecules | Murcko scaffolds: {len(scaffolds)} | Butina clusters: {len(clusters)}")
    print(
        f"GNN ({args.arch.upper()}): Train RMSE={report.train_metrics.rmse:.3f} | "
        f"Val RMSE={report.val_metrics.rmse:.3f} | Spearman={report.val_metrics.spearman_rho:.3f} | "
        f"Conformal Coverage={100.0 * cov:.1f}%"
    )
    if mmps:
        top = mmps[0]
        print(
            f"Top MMP Shift: {top.from_fragment} -> {top.to_fragment} "
            f"(delta {top.property_name} = {top.delta_property:+.2f})"
        )
    return 0


def _cmd_dock(args: argparse.Namespace) -> int:
    pocket = build_kinase_atp_pocket()
    pose = dock_ligand(args.smiles, pocket=pocket, mc_steps=args.steps, seed=args.seed)
    print(render_docking_profile(pose))
    return 0


def _cmd_protac(args: argparse.Namespace) -> int:
    base = default_protac_benchmark_specs()["optimal_rigid_linker"]
    linker = "C" * max(3, min(20, args.linker_length))
    spec = ProtacSpec(
        name=f"PROTAC_L{args.linker_length}",
        warhead_smiles=base.warhead_smiles,
        linker_smiles=linker if args.linker_length != 10 else base.linker_smiles,
        e3_ligand_smiles=base.e3_ligand_smiles,
        kd_poi_uM=base.kd_poi_uM,
        kd_e3_uM=base.kd_e3_uM,
        cooperativity_alpha=args.cooperativity,
    )
    deg = simulate_protac_degradation(spec, hours=args.hours)
    print(render_protac_hook_curve(deg))
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    traj = optimize_molecules_mcts_pareto(
        seed_smiles=args.seed_smiles,
        generations=args.generations,
        population_size=args.population,
        seed=args.seed,
    )
    print(render_pareto_scatter(traj))
    return 0


def _cmd_campaign(args: argparse.Namespace) -> int:
    report = run_full_discovery_campaign(
        architecture=args.arch,
        epochs=args.epochs,
        opt_generations=args.generations,
        seed=args.seed,
    )
    print(render_campaign_dashboard(report))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(campaign_report_to_dict(report), fh, indent=2)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the AlchemiDock argparse CLI parser."""
    parser = argparse.ArgumentParser(
        prog="alchemi-dock",
        description="Computational drug discovery, GNN QSAR, ADME/PK, docking, and PROTAC simulator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prof = sub.add_parser("profile", help="Parse SMILES and compute descriptors, ADME, and oral PK curve")
    p_prof.add_argument("--smiles", type=str, default="COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC")
    p_prof.add_argument("--dose-mg", type=float, default=100.0)
    p_prof.add_argument("--json", action="store_true")
    p_prof.set_defaults(func=_cmd_profile)

    p_scr = sub.add_parser("screen", help="Run Butina clustering, MMP mining, and scaffold-split GNN screening")
    p_scr.add_argument("--arch", type=str, default="gatv2", choices=["gcn", "mpnn", "gatv2"])
    p_scr.add_argument("--epochs", type=int, default=20)
    p_scr.add_argument("--cutoff", type=float, default=0.45)
    p_scr.add_argument("--seed", type=int, default=42)
    p_scr.set_defaults(func=_cmd_screen)

    p_dock = sub.add_parser("dock", help="Dock a ligand SMILES into the ATP kinase pocket")
    p_dock.add_argument("--smiles", type=str, default="Fc1ccc(Nc2ncnc3ccccc23)cc1")
    p_dock.add_argument("--steps", type=int, default=140)
    p_dock.add_argument("--seed", type=int, default=42)
    p_dock.set_defaults(func=_cmd_dock)

    p_prot = sub.add_parser("protac", help="Simulate PROTAC ternary equilibrium Hook curve and degradation")
    p_prot.add_argument("--cooperativity", type=float, default=None)
    p_prot.add_argument("--linker-length", type=int, default=10)
    p_prot.add_argument("--hours", type=float, default=24.0)
    p_prot.add_argument("--seed", type=int, default=42)
    p_prot.set_defaults(func=_cmd_protac)

    p_opt = sub.add_parser("optimize", help="Run MCTS + Pareto de novo molecular optimization")
    p_opt.add_argument("--seed-smiles", type=str, default="Fc1ccc(Nc2ncnc3ccccc23)cc1")
    p_opt.add_argument("--generations", type=int, default=6)
    p_opt.add_argument("--population", type=int, default=14)
    p_opt.add_argument("--seed", type=int, default=42)
    p_opt.set_defaults(func=_cmd_optimize)

    for cmd_name in ("benchmark", "demo"):
        p_cmp = sub.add_parser(cmd_name, help=f"Run full end-to-end {cmd_name} campaign")
        p_cmp.add_argument("--arch", type=str, default="gatv2", choices=["gcn", "mpnn", "gatv2"])
        p_cmp.add_argument("--epochs", type=int, default=20)
        p_cmp.add_argument("--generations", type=int, default=5)
        p_cmp.add_argument("--seed", type=int, default=42)
        p_cmp.add_argument("--json-out", type=str, default="")
        p_cmp.set_defaults(func=_cmd_campaign)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint returning process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
