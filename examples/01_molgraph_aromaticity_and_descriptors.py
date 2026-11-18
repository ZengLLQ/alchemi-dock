"""Example 1: SMILES graph perception, Huckel aromaticity, Gasteiger charges, and descriptors."""

from __future__ import annotations

from alchemi_dock import (
    bemis_murcko_scaffold,
    compute_clogp_and_mr,
    compute_fingerprint_set,
    compute_qed,
    compute_sa_score,
    compute_tpsa,
    parse_smiles,
    render_molecule_summary,
    tanimoto_similarity,
)


def main() -> None:
    erlotinib = "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC"
    gefitinib = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"

    mol_e = parse_smiles(erlotinib, name="Erlotinib")
    mol_g = parse_smiles(gefitinib, name="Gefitinib")

    print(render_molecule_summary(mol_e))
    clogp_e, mr_e = compute_clogp_and_mr(mol_e)
    print(
        f"\nErlotinib: cLogP={clogp_e:.2f}, MR={mr_e:.2f}, TPSA={compute_tpsa(mol_e):.1f}, "
        f"QED={compute_qed(mol_e):.3f}, SA={compute_sa_score(mol_e):.2f}"
    )

    scaff_e = bemis_murcko_scaffold(mol_e)
    scaff_g = bemis_murcko_scaffold(mol_g)
    fp_e = compute_fingerprint_set(mol_e)
    fp_g = compute_fingerprint_set(mol_g)
    sim = tanimoto_similarity(fp_e.ecfp4_bits, fp_g.ecfp4_bits)
    print(f"Erlotinib Murcko Scaffold : {scaff_e.murcko_smiles}")
    print(f"Gefitinib Murcko Scaffold : {scaff_g.murcko_smiles}")
    print(f"ECFP4 Tanimoto Similarity : {sim:.3f}")


if __name__ == "__main__":
    main()
