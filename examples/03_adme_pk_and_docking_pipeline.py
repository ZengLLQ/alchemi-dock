"""Example 3: Multitask ADME pharmacokinetics, oral PK ODE simulation, and ATP pocket docking."""

from __future__ import annotations

from alchemi_dock import (
    build_kinase_atp_pocket,
    dock_ligand,
    parse_smiles,
    predict_adme_profile,
    render_docking_profile,
    render_molecule_summary,
    render_pk_curve,
    simulate_oral_pk,
)


def main() -> None:
    lead_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC"
    mol = parse_smiles(lead_smiles, name="Quinazoline_Lead")
    adme = predict_adme_profile(lead_smiles)
    pk = simulate_oral_pk(adme, dose_mg=100.0, dosing_interval_h=12.0, n_doses=3, ec50_uM=0.10)

    print(render_molecule_summary(mol, adme=adme))
    print("")
    print(render_pk_curve(pk))

    pocket = build_kinase_atp_pocket()
    pose = dock_ligand(lead_smiles, pocket=pocket, mc_steps=140, seed=42)
    print("")
    print(render_docking_profile(pose))


if __name__ == "__main__":
    main()
