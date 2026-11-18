"""Multitask ADME pharmacokinetics, CYP450 enzyme kinetics, and oral PK ODE simulation.

Implements:
- Delaney ESOL aqueous solubility (logS)
- Caco-2 permeability (Papp) and blood-brain barrier (logBB / CNS MPO)
- Ionization microspecies classification (acid / base / zwitterion / neutral) and f_u,p
- Multi-isoform CYP3A4 / CYP2D6 / CYP2C9 Michaelis-Menten intrinsic clearance & MBI DDI
- Well-stirred and parallel-tube hepatic extraction (E_H, CL_H, F_H)
- hERG cardiotoxicity & CYP3A4 inhibition probability calibration
- Multi-dose 1-compartment oral PK ODE trajectory (Cmax, Tmax, AUC, t1/2)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .descriptors import (
    compute_clogp_and_mr,
    compute_fsp3,
    compute_qed,
    compute_sa_score,
    compute_tpsa,
    count_hbd_hba,
    count_lipinski_violations,
    detect_structural_alerts,
)
from .molgraph import identify_rotatable_bonds, match_smarts, molecular_weight, parse_smiles
from .types import ADMEProfile, MoleculeGraph, PKCurveResult


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def classify_ionization(mol: MoleculeGraph) -> Tuple[str, float]:
    """Classify dominant physiological pH 7.4 ionization microspecies and estimate pKa."""
    has_acid = bool(match_smarts(mol, "C(=O)[OH]")) or bool(match_smarts(mol, "S(=O)(=O)[NH]"))
    has_phenol = bool(match_smarts(mol, "c[OH]"))
    has_aliph_amine = any(
        a.element == "N"
        and not a.is_aromatic
        and a.formal_charge >= 0
        and not any(
            mol.atoms[n].element == "C"
            and any(mol.atoms[nn].element == "O" and bb.order >= 1.8 for nn, bb in mol.neighbor_bonds(n))
            for n in mol.neighbors(a.index)
        )
        for a in mol.atoms
    )
    if has_acid and has_aliph_amine:
        return "zwitterion", 6.8
    if has_acid:
        return "acid", 4.2
    if has_aliph_amine:
        return "base", 8.7
    if has_phenol:
        return "weak_acid", 9.6
    return "neutral", 7.0


def predict_esol_solubility(mol: MoleculeGraph, clogp: float, mw: float, rotb: int) -> float:
    """Delaney (2004) ESOL aqueous solubility logS (log10 mol/L)."""
    arom_atoms = sum(1 for a in mol.atoms if a.is_aromatic)
    ap = arom_atoms / max(1, mol.num_atoms)
    return float(0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rotb - 0.74 * ap)


def predict_caco2_permeability(clogp: float, tpsa: float, hbd: int, mw: float) -> float:
    """Predict Caco-2 apparent permeability Papp (10^-6 cm/s)."""
    log_papp = 1.25 - 0.011 * tpsa - 0.14 * hbd + 0.16 * clogp - 0.0012 * max(0.0, mw - 300.0)
    return float(max(0.05, min(65.0, 10.0 ** log_papp)))


def predict_bbb_and_cns_mpo(
    mw: float,
    clogp: float,
    tpsa: float,
    hbd: int,
    pka: float,
) -> Tuple[float, float]:
    """Predict blood-brain barrier logBB (Clark) and Pfizer CNS MPO score in [0, 6]."""
    log_bb = 0.152 * clogp - 0.0148 * tpsa + 0.139

    def _ramp(val: float, opt_hi: float, zero_at: float) -> float:
        if val <= opt_hi:
            return 1.0
        if val >= zero_at:
            return 0.0
        return (zero_at - val) / (zero_at - opt_hi)

    # Pfizer CNS MPO 6 components
    s_clogp = _ramp(clogp, 3.0, 5.0)
    logd74 = clogp - 0.6 if pka > 8.0 else (clogp - 1.5 if pka < 5.0 else clogp)
    s_logd = _ramp(logd74, 2.0, 4.0)
    s_mw = _ramp(mw, 360.0, 500.0)
    s_tpsa = 1.0 if 40.0 <= tpsa <= 90.0 else (_ramp(tpsa, 90.0, 120.0) if tpsa > 90.0 else max(0.0, tpsa / 40.0))
    s_hbd = _ramp(float(hbd), 0.5, 3.5)
    s_pka = _ramp(pka, 8.0, 10.0)
    cns_mpo = s_clogp + s_logd + s_mw + s_tpsa + s_hbd + s_pka
    return float(log_bb), float(cns_mpo)


def predict_plasma_unbound_fraction(clogp: float, ion_class: str) -> float:
    """Estimate fraction unbound in human plasma (f_u,p) in [0.002, 0.98]."""
    shift = 0.55 if ion_class == "acid" else (-0.25 if ion_class == "base" else 0.0)
    log_fu = -0.35 - 0.34 * max(-1.0, clogp) - shift
    return float(max(0.002, min(0.98, 10.0 ** log_fu)))


def cyp_michaelis_menten_clearance(
    clogp: float,
    mw: float,
    tpsa: float,
    ion_class: str,
    substrate_uM: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute multi-isoform CYP3A4/2D6/2C9 Michaelis-Menten intrinsic clearance & DDI risk.

    Returns (CL_int [uL/min/mg], CYP3A4_inhibition_prob, DDI_AUC_ratio).
    """
    # Lipophilicity and aromaticity drive CYP3A4 affinity; basicity drives CYP2D6; acidity drives CYP2C9
    vmax_3a4 = 180.0 * _sigmoid(0.75 * (clogp - 2.2) + 0.004 * (mw - 320.0))
    km_3a4 = max(1.5, 25.0 * math.exp(-0.42 * max(-1.0, clogp)))

    vmax_2d6 = 85.0 * (1.4 if ion_class == "base" else 0.35) * _sigmoid(0.5 * (clogp - 1.5))
    km_2d6 = 8.0 if ion_class == "base" else 35.0

    vmax_2c9 = 70.0 * (1.5 if ion_class in ("acid", "weak_acid") else 0.30)
    km_2c9 = 10.0 if ion_class in ("acid", "weak_acid") else 40.0

    cl_int = (
        vmax_3a4 / (km_3a4 + substrate_uM)
        + vmax_2d6 / (km_2d6 + substrate_uM)
        + vmax_2c9 / (km_2c9 + substrate_uM)
    )
    cl_int *= max(0.35, 1.0 - 0.0025 * max(0.0, tpsa - 60.0))

    cyp3a4_inh = _sigmoid(-1.8 + 0.65 * clogp + 0.003 * (mw - 320.0) - 0.014 * tpsa)
    # Static mechanistic DDI AUC ratio: 1 + [I]/Ki + MBI contribution
    ki_est_uM = max(0.25, 18.0 * math.exp(-0.55 * max(0.0, clogp)))
    ddi_ratio = 1.0 + (substrate_uM / ki_est_uM) * (0.4 + 0.8 * cyp3a4_inh)
    return float(cl_int), float(cyp3a4_inh), float(ddi_ratio)


def hepatic_extraction_well_stirred(
    cl_int_ul_min_mg: float,
    fu_plasma: float,
    blood_flow_L_h: float = 87.0,
    mppgl_mg_g: float = 40.0,
    liver_mass_g: float = 1500.0,
) -> Tuple[float, float, float]:
    """Scale microsomal CL_int to human hepatic extraction E_H, CL_H (L/h), and F_H."""
    # Scale uL/min/mg -> L/h whole liver
    cl_int_whole_L_h = cl_int_ul_min_mg * mppgl_mg_g * liver_mass_g * 1e-6 * 60.0
    fu_cl = fu_plasma * cl_int_whole_L_h
    cl_h = (blood_flow_L_h * fu_cl) / max(1e-9, blood_flow_L_h + fu_cl)
    eh = cl_h / blood_flow_L_h
    fh = max(0.01, 1.0 - eh)
    return float(eh), float(cl_h), float(fh)


def predict_herg_risk(mol: MoleculeGraph, clogp: float, ion_class: str, tpsa: float) -> float:
    """Predict hERG potassium-channel blockade probability in [0, 1]."""
    arom_count = sum(1 for cyc in mol.sssr_rings if all(mol.atoms[i].is_aromatic for i in cyc))
    basic_boost = 1.35 if ion_class == "base" else 0.0
    logit = -3.1 + 0.68 * clogp + 0.55 * arom_count + basic_boost - 0.018 * tpsa
    return float(_sigmoid(logit))


def predict_adme_profile(smiles: str) -> ADMEProfile:
    """Run full multitask ADME, physicochemical, CYP, and safety prediction on SMILES."""
    mol = parse_smiles(smiles)
    mw = molecular_weight(mol)
    clogp, _ = compute_clogp_and_mr(mol)
    tpsa = compute_tpsa(mol)
    hbd, hba = count_hbd_hba(mol)
    rotb = len(identify_rotatable_bonds(mol))
    fsp3 = compute_fsp3(mol)
    alerts = detect_structural_alerts(mol)
    qed = compute_qed(mol, alerts_count=len(alerts))
    sa = compute_sa_score(mol)
    ro5 = count_lipinski_violations(mw, clogp, hbd, hba)
    ion_class, pka = classify_ionization(mol)

    log_s = predict_esol_solubility(mol, clogp, mw, rotb)
    papp = predict_caco2_permeability(clogp, tpsa, hbd, mw)
    log_bb, cns_mpo = predict_bbb_and_cns_mpo(mw, clogp, tpsa, hbd, pka)
    fu = predict_plasma_unbound_fraction(clogp, ion_class)
    cl_int, cyp3a4_inh, ddi_ratio = cyp_michaelis_menten_clearance(clogp, mw, tpsa, ion_class)
    eh, _, fh = hepatic_extraction_well_stirred(cl_int, fu)

    # Oral bioavailability F = Fa * Fg * Fh
    fa = min(0.98, 1.0 - math.exp(-0.18 * papp))
    fg = max(0.35, 1.0 - 0.35 * cyp3a4_inh)
    oral_f = max(0.01, min(0.98, fa * fg * fh))
    herg_prob = predict_herg_risk(mol, clogp, ion_class, tpsa)

    return ADMEProfile(
        smiles=mol.canonical_smiles or smiles,
        mw=mw,
        clogp=clogp,
        tpsa=tpsa,
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rotb,
        fsp3=fsp3,
        qed=qed,
        sa_score=sa,
        ro5_violations=ro5,
        alerts=alerts,
        log_s_esol=log_s,
        papp_caco2_1e6_cms=papp,
        log_bb=log_bb,
        cns_mpo=cns_mpo,
        fu_plasma=fu,
        cl_int_ul_min_mg=cl_int,
        hepatic_extraction_eh=eh,
        oral_bioavailability_f=oral_f,
        herg_risk_prob=herg_prob,
        cyp3a4_inhibition_prob=cyp3a4_inh,
        cyp_ddi_auc_ratio=ddi_ratio,
        ionization_class=ion_class,
        estimated_pka=pka,
    )


def simulate_oral_pk(
    adme: ADMEProfile,
    dose_mg: float = 100.0,
    dosing_interval_h: float = 12.0,
    n_doses: int = 3,
    ec50_uM: float = 0.15,
    vd_L: float = 45.0,
) -> PKCurveResult:
    """Simulate a multi-dose 1-compartment oral pharmacokinetic ODE trajectory."""
    _, cl_h, _ = hepatic_extraction_well_stirred(adme.cl_int_ul_min_mg, adme.fu_plasma)
    ke = cl_h / max(1.0, vd_L)  # elimination rate constant (1/h)
    ka = max(0.25, min(2.8, 0.25 + 0.08 * adme.papp_caco2_1e6_cms))
    dose_umol = (dose_mg * 1e-3 / max(50.0, adme.mw)) * 1e6  # micromoles
    bioavail = adme.oral_bioavailability_f

    total_hours = dosing_interval_h * n_doses + 12.0
    dt_eval = np.linspace(0.0, total_hours, 181)
    conc_uM = np.zeros_like(dt_eval)

    # Integrate segment by segment between oral bolus doses for exact conservation
    gut_umol = 0.0
    plasma_umol = 0.0
    dose_times = [k * dosing_interval_h for k in range(n_doses)] + [total_hours]

    def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
        a_gut, a_plasma = y[0], y[1]
        return np.array([-ka * a_gut, ka * bioavail * a_gut - ke * a_plasma], dtype=np.float64)

    for idx in range(len(dose_times) - 1):
        t0 = dose_times[idx]
        t1 = dose_times[idx + 1]
        gut_umol += dose_umol
        mask = (dt_eval >= t0 - 1e-9) & (dt_eval <= t1 + 1e-9)
        t_sub = dt_eval[mask]
        if len(t_sub) == 0:
            t_sub = np.array([t0, t1], dtype=np.float64)
        sol = solve_ivp(
            _rhs,
            (t0, t1),
            np.array([gut_umol, plasma_umol], dtype=np.float64),
            t_eval=t_sub,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        # Re-evaluate endpoints cleanly for continuity
        sol_end = solve_ivp(
            _rhs,
            (t0, t1),
            np.array([gut_umol, plasma_umol], dtype=np.float64),
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        gut_umol = float(sol_end.y[0, -1])
        plasma_umol = float(sol_end.y[1, -1])
        if np.any(mask):
            conc_uM[mask] = sol.y[1] / vd_L

    cmax = float(np.max(conc_uM))
    tmax = float(dt_eval[int(np.argmax(conc_uM))])
    auc = float(np.sum(0.5 * (conc_uM[:-1] + conc_uM[1:]) * np.diff(dt_eval)))
    half_life = math.log(2.0) / max(1e-6, ke)
    margin = cmax / max(1e-6, ec50_uM)
    return PKCurveResult(
        times_h=dt_eval,
        plasma_conc_uM=conc_uM,
        cmax_uM=cmax,
        tmax_h=tmax,
        auc_0_inf_uM_h=auc,
        half_life_h=half_life,
        clearance_L_h=cl_h,
        therapeutic_index_margin=float(margin),
    )
