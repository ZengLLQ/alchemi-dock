"""Example 2: Scaffold-split GNN QSAR modeling (GCN/MPNN/GATv2), MMP mining, and Conformal UQ."""

from __future__ import annotations

from alchemi_dock import (
    MolecularGNN,
    butina_clustering,
    compute_fingerprint_set,
    conformal_predict_intervals,
    empirical_conformal_coverage,
    featurize_mol_tensor,
    load_kinase_benchmark_dataset,
    mine_matched_molecular_pairs,
    parse_smiles,
    scaffold_split_indices,
    split_conformal_quantile,
)


def main() -> None:
    names, smiles_list, targets = load_kinase_benchmark_dataset(seed=42)
    mols = [parse_smiles(s, name=n) for n, s in zip(names, smiles_list)]
    fps = [compute_fingerprint_set(m).ecfp4_bits for m in mols]
    clusters = butina_clustering(fps, similarity_cutoff=0.45)
    mmps = mine_matched_molecular_pairs(smiles_list, targets.tolist(), property_name="pIC50")

    print(f"Loaded {len(mols)} ligands into {len(clusters)} Butina clusters and {len(mmps)} MMPs.")
    if mmps:
        mmp0 = mmps[0]
        print(f"Strongest MMP: {mmp0.from_fragment} -> {mmp0.to_fragment} (d_pIC50={mmp0.delta_property:+.2f})")

    graphs = [featurize_mol_tensor(m) for m in mols]
    tr_idx, va_idx = scaffold_split_indices(smiles_list, val_fraction=0.30)

    for arch in ("gcn", "mpnn", "gatv2"):
        model = MolecularGNN(architecture=arch, hidden_dim=28, num_layers=2, num_heads=2, seed=42)
        report = model.fit(
            [graphs[i] for i in tr_idx],
            targets[tr_idx],
            [graphs[i] for i in va_idx],
            targets[va_idx],
            epochs=20,
            lr=8e-3,
        )
        preds, _ = model.predict([graphs[i] for i in va_idx])
        mid = max(1, len(preds) // 2)
        q_rad = split_conformal_quantile(targets[va_idx][:mid], preds[:mid], alpha=0.15)
        ivs = conformal_predict_intervals(preds[mid:], q_rad, alpha=0.15)
        cov = empirical_conformal_coverage(targets[va_idx][mid:], ivs)
        print(
            f"[{arch.upper():<5s}] Train RMSE={report.train_metrics.rmse:.3f} | "
            f"Val RMSE={report.val_metrics.rmse:.3f} | Spearman={report.val_metrics.spearman_rho:.3f} | "
            f"Conformal Coverage={100.0 * cov:.1f}% (radius=+/-{q_rad:.2f})"
        )


if __name__ == "__main__":
    main()
