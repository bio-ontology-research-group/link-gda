#!/usr/bin/env python3
"""
Stepwise Bias Evaluation (v3):
Evaluates all original models, matched 10-fold refitted Phrank and PhenoRank variants
(including 1,000-simulation empirical p-values per fold), all 9 retrained ULTRA/PhenoQuery
arms, and (when present) all retrained no-bridge LinkGDA, INDIGENA, and ConvKB-D arms
across all 10 folds (fold0..9).
"""

import json
import pathlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = pathlib.Path("/mnt/data1/agents/projects/ultra-top-hits-20261023")
MATCHED_ROOT = pathlib.Path("/mnt/data1/agents/projects/ultra-matched-20261027")
EQ_ROOT = pathlib.Path("/mnt/data1/agents/projects/ultra-eq-query-20261026")
BC_V3_ROOT = ROOT / "baselines/corrected_v3"
NOBRIDGE_ROOT = ROOT / "nobridge_runs"
LOCAL_NPZ = ROOT / "comparators/local-npz"


def rank_1d_with_ties(scores_1d, target_idx, mask=None):
    if mask is not None:
        m = mask.copy()
        m[target_idx] = True
        sub = scores_1d[m]
        val = scores_1d[target_idx]
        return 1.0 + float(np.sum(sub > val)) + 0.5 * float(np.sum(sub == val) - 1)
    val = scores_1d[target_idx]
    return 1.0 + float(np.sum(scores_1d > val)) + 0.5 * float(np.sum(scores_1d == val) - 1)


def std_loo_cal(scores):
    x0 = scores.astype(np.float64) - scores[:1]
    n = len(x0) - 1
    mu = (x0.sum(0) - x0) / n
    var = (np.square(x0).sum(0) - np.square(x0)) / n - mu**2
    sd = np.sqrt(np.maximum(var, 0.0))
    return (x0 - mu) / (sd + 1e-12)


def dedup_loo_cal(scores, diseases, profiles):
    n_d, _ = scores.shape
    prof_tuples = [tuple(sorted(set(profiles[d]))) for d in diseases]
    rep_indices = []
    seen_profs = {}
    d_to_rep = np.empty(n_d, dtype=int)
    for i, pt in enumerate(prof_tuples):
        if pt not in seen_profs:
            seen_profs[pt] = len(rep_indices)
            rep_indices.append(i)
        d_to_rep[i] = seen_profs[pt]
    rep_scores = scores[rep_indices].astype(np.float64)
    x0 = rep_scores - rep_scores[:1]
    n_rep = len(rep_indices) - 1
    tot_sum = x0.sum(0)
    tot_sq = np.square(x0).sum(0)

    out = np.empty_like(scores, dtype=np.float64)
    for i in range(n_d):
        r_i = d_to_rep[i]
        mu = (tot_sum - x0[r_i]) / n_rep
        var = (tot_sq - np.square(x0[r_i])) / n_rep - np.square(mu)
        sd = np.sqrt(np.maximum(var, 0.0))
        out[i] = ((scores[i].astype(np.float64) - rep_scores[0]) - mu) / (sd + 1e-12)
    return out


def status_z_standardize(cal_scores, cand_seen):
    out = np.empty_like(cal_scores, dtype=np.float64)
    s_idx = np.flatnonzero(cand_seen)
    u_idx = np.flatnonzero(~cand_seen)
    s_sub = cal_scores[:, s_idx]
    u_sub = cal_scores[:, u_idx]
    s_mu = np.mean(s_sub, axis=1, keepdims=True)
    s_sd = np.std(s_sub, axis=1, keepdims=True)
    u_mu = np.mean(u_sub, axis=1, keepdims=True)
    u_sd = np.std(u_sub, axis=1, keepdims=True)
    out[:, s_idx] = (s_sub - s_mu) / (s_sd + 1e-12)
    out[:, u_idx] = (u_sub - u_mu) / (u_sd + 1e-12)
    return out


def get_fold_meta(f):
    fdir = ROOT / f"data/fold{f}"
    profiles = json.loads((fdir / "profiles.json").read_text())
    train_pairs = [
        tuple(l.strip().split("\t")) for l in (fdir / "training.tsv").read_text().splitlines()[1:]
    ]
    test_pairs = [
        tuple(l.strip().split("\t")) for l in (fdir / "test.tsv").read_text().splitlines()[1:]
    ]
    genes = (fdir / "candidates.txt").read_text().splitlines()
    diseases = sorted({d for _, d in test_pairs})
    return f, genes, diseases, train_pairs, test_pairs, profiles


def iter_models_for_fold(f, genes, diseases, train_pairs, profiles):
    gi = {g: i for i, g in enumerate(genes)}

    def load_npz(path, key="scores"):
        with np.load(path) as z:
            s = z[key].astype(np.float64)
            ds = z["diseases"].astype(str).tolist()
            gs = z["genes"].astype(str).tolist()
        if gs != genes:
            g_map = [gs.index(g) for g in genes]
            s = s[:, g_map]
        if ds != diseases:
            d_map = [ds.index(d) for d in diseases]
            s = s[d_map]
        return s

    # 1. Local comparators (original graph with disease bridges)
    for name, folder in [
        ("LinkGDA-U1 (PFS)", "main-linkgda-pfs-calsel"),
        ("LinkGDA-U1 (PF, No Expr)", "main-linkgda-pf-calsel"),
        ("LinkGDA-U1 (F, No Mouse Pheno)", "main-linkgda-f-calsel"),
        ("ConvKB-D (PFS)", "main-convkb-pfs-calsel"),
        ("INDIGENA (PFS)", "main-indigena-pfs-calsel"),
        ("ULTRA (Zero-Shot BMA)", "main-ultra-pfs"),
        ("Resnik-BMA", "resnik_resnik_bma"),
        ("Lin-BMA", "resnik_lin_bma"),
        ("SimGIC", "resnik_simgic"),
    ]:
        yield name, load_npz(LOCAL_NPZ / folder / f"fold_{f}.npz")

    # 2. TransD diagnostic (T1 & P-pool on original graph)
    yield "LinkGDA-T1 (Typed Neg)", load_npz(
        ROOT / f"transd-diagnostic/runs/score-fold{f}/T1_calibrated_head/test/scores.npz"
    )
    yield "LinkGDA-Ppool (Profile Hard-Neg)", load_npz(
        ROOT / f"transd-diagnostic/runs/score-fold{f}/P-pool_raw_head/test/scores.npz"
    )

    # 3. ULTRA mechanism interventions (inference-time graph surgeries)
    yield "ULTRA (Control Mean-Pool)", load_npz(
        ROOT / f"transd-ultra-mechanism/gpu/fold{f}/control/scores.npz"
    )
    yield "ULTRA (No Disease Bridges)", load_npz(
        ROOT / f"transd-ultra-mechanism/gpu/fold{f}/no_disease_bridges/scores.npz"
    )
    yield "ULTRA (No Expression Hubs)", load_npz(
        ROOT / f"transd-ultra-mechanism/gpu/fold{f}/no_expression/scores.npz"
    )

    # ULTRA masked direct causes_phenotype from test-phenotypes.npz
    g_train_p = defaultdict(set)
    for g, d in train_pairs:
        g_train_p[g].update(profiles[d])
    with np.load(ROOT / f"phrank-inspired/ultra-cache/fold{f}/test-phenotypes.npz") as zp:
        ps_mat = zp["scores"].astype(np.float64)
        p_list = zp["phenotypes"].astype(str).tolist()
    pi = {p: i for i, p in enumerate(p_list)}
    direct_mask = np.zeros_like(ps_mat, dtype=bool)
    for g, ps in g_train_p.items():
        if g in gi:
            g_i = gi[g]
            for p in ps:
                if p in pi:
                    direct_mask[pi[p], g_i] = True
    non_direct_cnt = np.maximum(np.sum(~direct_mask, axis=0), 1)
    non_direct_mean = np.sum(np.where(~direct_mask, ps_mat, 0.0), axis=0) / non_direct_cnt
    masked_d_scores = np.empty((len(diseases), len(genes)), dtype=np.float64)
    for i, d in enumerate(diseases):
        p_idxs = [pi[p] for p in profiles[d]]
        sub = ps_mat[p_idxs]
        sub_m = direct_mask[p_idxs]
        v_cnt = np.sum(~sub_m, axis=0)
        s_val = np.sum(np.where(~sub_m, sub, 0.0), axis=0)
        masked_d_scores[i] = np.where(v_cnt > 0, s_val / np.maximum(v_cnt, 1), non_direct_mean)
    del ps_mat, direct_mask
    yield "ULTRA (Direct Overlap Masked)", masked_d_scores

    # 4. Matched ULTRA & UltraQuery (with disease nodes)
    for co, prefix in [("general", "ULTRA-Matched"), ("query", "UltraQuery-Matched")]:
        for arm in [
            "original_pool_frozen",
            "original_pool_adapted",
            "augmented_pool_adapted",
            "augmented_and_adapted",
        ]:
            yield f"{prefix} ({arm})", load_npz(
                MATCHED_ROOT / co / f"score-fold{f}" / arm / "scores.npz"
            )

    # 5. All 9 retrained PhenoQuery arms (End-to-End Retrained on No-Disease Graph vs Bridge Graph)
    for arm in [
        "bridge_pool",
        "plain_pool",
        "plain_and",
        "entity_pool",
        "entity_and",
        "typed_pool",
        "typed_and",
        "structure_pool",
        "structure_and",
    ]:
        p_arm = EQ_ROOT / f"test/fold{f}/{arm}/scores.npz"
        if p_arm.exists():
            yield f"PhenoQuery ({arm})", load_npz(p_arm)

    # 6. Matched 10-Fold Refitted Phrank & PhenoRank (corrected_v3)
    v3_fdir = BC_V3_ROOT / f"fold{f}"
    for fname, label in [
        ("phrank_hpo_disease_raw_scores.npz", "Phrank-HPO (disease_raw)"),
        ("phrank_hpo_disease_norm_scores.npz", "Phrank-HPO (disease_normalized)"),
        ("phrank_hpo_direct_raw_scores.npz", "Phrank-HPO (direct_raw)"),
        ("phrank_hpo_direct_norm_scores.npz", "Phrank-HPO (direct_normalized)"),
        ("phrank_upheno_hm_disease_raw_scores.npz", "Phrank-UPheno (Human+MGI Disease Raw)"),
        ("phrank_upheno_hm_disease_norm_scores.npz", "Phrank-UPheno (Human+MGI Disease Norm)"),
        ("phrank_upheno_hm_direct_raw_scores.npz", "Phrank-UPheno (Human+MGI Direct Raw)"),
        ("phrank_upheno_hm_direct_norm_scores.npz", "Phrank-UPheno (Human+MGI Direct Norm)"),
        ("phrank_upheno_mgi_raw_scores.npz", "Phrank-UPheno (MGI-Only Raw)"),
        ("phrank_upheno_mgi_norm_scores.npz", "Phrank-UPheno (MGI-Only Norm)"),
        ("phenorank_upheno_hm_add_nosim_scores.npz", "PhenoRank-UPheno (HM-Add+PPI NoSim)"),
        ("phenorank_upheno_hm_add_sim_scores.npz", "PhenoRank-UPheno (HM-Add+PPI 1000-Sim)"),
        ("phenorank_upheno_hm_add_sim_tb_scores.npz", "PhenoRank-UPheno (HM-Add+PPI 1000-Sim TB)"),
        ("phenorank_upheno_hm_bal_nosim_scores.npz", "PhenoRank-UPheno (HM-Bal+PPI NoSim)"),
        ("phenorank_upheno_hm_bal_sim_scores.npz", "PhenoRank-UPheno (HM-Bal+PPI 1000-Sim)"),
        ("phenorank_upheno_hm_bal_sim_tb_scores.npz", "PhenoRank-UPheno (HM-Bal+PPI 1000-Sim TB)"),
        ("phenorank_upheno_mgi_ppi_nosim_scores.npz", "PhenoRank-UPheno (Mouse+PPI NoSim)"),
        ("phenorank_upheno_mgi_ppi_sim_scores.npz", "PhenoRank-UPheno (Mouse+PPI 1000-Sim)"),
        ("phenorank_upheno_mgi_ppi_sim_tb_scores.npz", "PhenoRank-UPheno (Mouse+PPI 1000-Sim TB)"),
        ("phenorank_upheno_mgi_noppi_nosim_scores.npz", "PhenoRank-UPheno (Mouse-NoPPI NoSim)"),
        ("phenorank_upheno_mgi_noppi_sim_scores.npz", "PhenoRank-UPheno (Mouse-NoPPI 1000-Sim)"),
        ("phenorank_upheno_mgi_noppi_sim_tb_scores.npz", "PhenoRank-UPheno (Mouse-NoPPI 1000-Sim TB)"),
        ("phenorank_2016_sim_tb_scores.npz", "PhenoRank (Static 2016 1000-Sim TB)"),
    ]:
        p = v3_fdir / fname
        if p.exists():
            yield label, load_npz(p)

    # Also keep legacy Static 2016 strict 1000-Sim for comparison
    p_leg = ROOT / f"baselines/phenorank/fold{f}/scores.npz"
    if p_leg.exists():
        yield "PhenoRank (Static 2016 1000-Sim)", load_npz(p_leg)

    # 7. Retrained No-Bridge KGE models (LinkGDA, INDIGENA, ConvKB-D) when all 10 folds exist
    nb_fdir = NOBRIDGE_ROOT / f"fold{f}"
    for fname, label in [
        ("linkgda_u1_nobridge_cal_mr_scores.npz", "LinkGDA-U1 (Retrained No-Bridge, Cal-MR)"),
        ("linkgda_u1_nobridge_raw_head_scores.npz", "LinkGDA-U1 (Retrained No-Bridge, Raw-Head)"),
        ("linkgda_t1_nobridge_cal_head_scores.npz", "LinkGDA-T1 (Retrained No-Bridge, Cal-Head)"),
        ("linkgda_t1_nobridge_raw_head_scores.npz", "LinkGDA-T1 (Retrained No-Bridge, Raw-Head)"),
        ("linkgda_ppool_nobridge_scores.npz", "LinkGDA-Ppool (Retrained No-Bridge)"),
        ("indigena_nobridge_raw_mr_scores.npz", "INDIGENA (Retrained No-Bridge, Raw-MR)"),
        ("indigena_nobridge_raw_head_scores.npz", "INDIGENA (Retrained No-Bridge, Raw-Head)"),
        ("convkbd_nobridge_cal_mr_scores.npz", "ConvKB-D (Retrained No-Bridge, Cal-MR)"),
        ("convkbd_nobridge_raw_mr_scores.npz", "ConvKB-D (Retrained No-Bridge, Raw-MR)"),
    ]:
        if all((NOBRIDGE_ROOT / f"fold{k}" / fname).exists() for k in range(10)):
            yield label, load_npz(nb_fdir / fname)


def eval_fold_v3(f):
    f, genes, diseases, train_pairs, test_pairs, profiles = get_fold_meta(f)
    di = {d: i for i, d in enumerate(diseases)}
    gi = {g: i for i, g in enumerate(genes)}
    seen_set = {g for g, _ in train_pairs}
    cand_seen = np.array([g in seen_set for g in genes], dtype=bool)

    g_train_p = defaultdict(set)
    for g, d in train_pairs:
        g_train_p[g].update(profiles[d])
    train_d_profs = {tuple(sorted(set(profiles[d]))) for _, d in train_pairs}

    pos_per_d = defaultdict(set)
    for g, d in test_pairs:
        pos_per_d[d].add(gi[g])

    pair_meta = []
    for g, d in test_pairs:
        pset = set(profiles[d])
        ptuple = tuple(sorted(pset))
        is_seen = g in seen_set
        j_same_gt0 = is_seen and (len(pset & g_train_p.get(g, set())) > 0)
        is_dup_train_prof = ptuple in train_d_profs
        pair_meta.append(
            {
                "g": g,
                "d": d,
                "gi": gi[g],
                "di": di[d],
                "seen": is_seen,
                "j_same_0": not j_same_gt0,
                "not_dup_prof": not is_dup_train_prof,
                "prof_tuple": ptuple,
                "multi_gene": len(pos_per_d[d]) > 1,
            }
        )

    fold_out = {}
    for mname, raw_scores in iter_models_for_fold(f, genes, diseases, train_pairs, profiles):
        cal_s1 = std_loo_cal(raw_scores)
        s2_scores = status_z_standardize(cal_s1, cand_seen)
        s2_raw_scores = status_z_standardize(raw_scores, cand_seen)
        cal_dedup = dedup_loo_cal(raw_scores, diseases, profiles)
        s4_scores = status_z_standardize(cal_dedup, cand_seen)

        r_s0, r_s1, r_s2, r_s4_filt = [], [], [], []
        r_s2_raw, r_s4_raw_filt = [], []
        r_s1_filt, r_s0_filt = [], []
        r_ws_s1, r_ws_s0 = [], []
        for pm in pair_meta:
            d_i = pm["di"]
            g_i = pm["gi"]
            d = pm["d"]
            r_s0.append(rank_1d_with_ties(raw_scores[d_i], g_i))
            r_s1.append(rank_1d_with_ties(cal_s1[d_i], g_i))
            r_s2.append(rank_1d_with_ties(s2_scores[d_i], g_i))
            r_s2_raw.append(rank_1d_with_ties(s2_raw_scores[d_i], g_i))

            if len(pos_per_d[d]) > 1:
                fmask = np.ones(len(genes), dtype=bool)
                for other_gi in pos_per_d[d]:
                    if other_gi != g_i:
                        fmask[other_gi] = False
                r_s4_filt.append(rank_1d_with_ties(s4_scores[d_i], g_i, mask=fmask))
                r_s4_raw_filt.append(rank_1d_with_ties(s2_raw_scores[d_i], g_i, mask=fmask))
                r_s1_filt.append(rank_1d_with_ties(cal_dedup[d_i], g_i, mask=fmask))
                r_s0_filt.append(rank_1d_with_ties(raw_scores[d_i], g_i, mask=fmask))
            else:
                r_s4_filt.append(rank_1d_with_ties(s4_scores[d_i], g_i))
                r_s4_raw_filt.append(rank_1d_with_ties(s2_raw_scores[d_i], g_i))
                r_s1_filt.append(rank_1d_with_ties(cal_dedup[d_i], g_i))
                r_s0_filt.append(rank_1d_with_ties(raw_scores[d_i], g_i))

            same_status_mask = cand_seen == pm["seen"]
            r_ws_s1.append(rank_1d_with_ties(cal_s1[d_i], g_i, mask=same_status_mask))
            r_ws_s0.append(rank_1d_with_ties(raw_scores[d_i], g_i, mask=same_status_mask))

        fold_out[mname] = {
            "r_s0": r_s0,
            "r_s1": r_s1,
            "r_s2": r_s2,
            "r_s2_raw": r_s2_raw,
            "r_s4_filt": r_s4_filt,
            "r_s4_raw_filt": r_s4_raw_filt,
            "r_s1_filt": r_s1_filt,
            "r_s0_filt": r_s0_filt,
            "r_ws_s1": r_ws_s1,
            "r_ws_s0": r_ws_s0,
        }
    return f, pair_meta, fold_out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate stepwise bias decomposition and corrections (S0 -> S4) across all models and baselines.")
    ap.add_argument("--base-dir", type=pathlib.Path, default=ROOT, help="Base workspace directory containing data/, baselines/, and nobridge_runs/")
    ap.add_argument("--eq-dir", type=pathlib.Path, default=EQ_ROOT, help="Directory containing PhenoQuery / EQ-UltraQuery fold outputs")
    ap.add_argument("--out-json", type=pathlib.Path, default=pathlib.Path("/tmp/stepwise_bias_report_v3.json"), help="Output JSON report path")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel fold workers")
    args = ap.parse_args()

    ROOT = args.base_dir
    EQ_ROOT = args.eq_dir
    BC_V3_ROOT = ROOT / "baselines/corrected_v3"
    NOBRIDGE_ROOT = ROOT / "nobridge_runs"
    LOCAL_NPZ = ROOT / "comparators/local-npz"

    all_meta = []
    all_models = defaultdict(lambda: defaultdict(list))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f, pmeta, fout in ex.map(eval_fold_v3, range(10)):
            all_meta.extend(pmeta)
            for mname, d in fout.items():
                for k, v in d.items():
                    all_models[mname][k].extend(v)
            print(f"Evaluated fold {f} ({len(fout)} models)", flush=True)

    # Include Exomiser-Phive from precomputed JSONs
    phive_s0, phive_s1, phive_ws = [], [], []
    for f in range(10):
        obj = json.loads((LOCAL_NPZ / "phive-json" / f"fold_{f}.json").read_text())
        phive_s0.extend(obj["settings"]["raw"]["ranks"])
        phive_s1.extend(obj["settings"]["disease_loo"]["ranks"])
        ws_d = obj["settings"]["disease_loo"]["within_training_status"]
        ws_map = {}
        s_n = ws_d["seen"]["unfiltered"]["candidates"]
        u_n = ws_d["unseen"]["unfiltered"]["candidates"]
        for idx, rk in zip(ws_d["seen"]["row_indices"], ws_d["seen"]["ranks"]):
            ws_map[idx] = (rk, s_n)
        for idx, rk in zip(ws_d["unseen"]["row_indices"], ws_d["unseen"]["ranks"]):
            ws_map[idx] = (rk, u_n)
        for i in range(len(obj["keys"])):
            rk, _ = ws_map[i]
            phive_ws.append(rk)
    all_models["Exomiser-Phive"]["r_s0"] = phive_s0
    all_models["Exomiser-Phive"]["r_s1"] = phive_s1
    all_models["Exomiser-Phive"]["r_s2"] = [
        1.0 + (rk - 1.0) * (13624.0 / (3778.2 if m["seen"] else 9844.8))
        for rk, m in zip(phive_ws, all_meta)
    ]
    all_models["Exomiser-Phive"]["r_s2_raw"] = all_models["Exomiser-Phive"]["r_s2"]
    all_models["Exomiser-Phive"]["r_s4_filt"] = all_models["Exomiser-Phive"]["r_s2"]
    all_models["Exomiser-Phive"]["r_s4_raw_filt"] = all_models["Exomiser-Phive"]["r_s2"]
    all_models["Exomiser-Phive"]["r_s1_filt"] = phive_s1
    all_models["Exomiser-Phive"]["r_s0_filt"] = phive_s0
    all_models["Exomiser-Phive"]["r_ws_s1"] = phive_ws
    all_models["Exomiser-Phive"]["r_ws_s0"] = phive_ws

    seen_mask = np.array([m["seen"] for m in all_meta], dtype=bool)
    j0_mask = np.array([m["j_same_0"] for m in all_meta], dtype=bool)
    nodup_mask = np.array([m["not_dup_prof"] for m in all_meta], dtype=bool)
    s4_mask = j0_mask & nodup_mask

    def stats(arr, mask=None, equal_disease=False):
        a = np.asarray(arr, dtype=np.float64)
        if mask is not None:
            a = a[mask]
            metas = [m for m, keep in zip(all_meta, mask) if keep]
        else:
            metas = all_meta
        if equal_disease:
            by_d = defaultdict(list)
            for val, m in zip(a, metas):
                by_d[m["prof_tuple"]].append(val)
            d_mr = np.mean([np.mean(v) for v in by_d.values()])
            d_mrr = np.mean([np.mean(1.0 / np.asarray(v)) for v in by_d.values()])
            d_h1 = np.mean([np.mean(np.asarray(v) <= 1.0) for v in by_d.values()]) * 100.0
            d_h10 = np.mean([np.mean(np.asarray(v) <= 10.0) for v in by_d.values()]) * 100.0
            d_med = float(np.median([np.mean(v) for v in by_d.values()]))
            return {
                "h1": float(d_h1),
                "h10": float(d_h10),
                "mrr": float(d_mrr),
                "mr": float(d_mr),
                "med": d_med,
                "n": len(by_d),
            }
        return {
            "h1": float(np.mean(a <= 1.0) * 100.0),
            "h10": float(np.mean(a <= 10.0) * 100.0),
            "mrr": float(np.mean(1.0 / a)),
            "mr": float(np.mean(a)),
            "med": float(np.median(a)),
            "n": len(a),
        }

    summary = {}
    for mname, d in all_models.items():
        summary[mname] = {
            "S0_Raw_All": stats(d["r_s0"]),
            "S0_Raw_Seen": stats(d["r_s0"], seen_mask),
            "S0_Raw_Unseen": stats(d["r_s0"], ~seen_mask),
            "S0_Raw_Seen_Jgt0": stats(d["r_s0"], seen_mask & (~j0_mask)),
            "S0_Raw_Seen_J0": stats(d["r_s0"], seen_mask & j0_mask),
            "S1_Cal_All": stats(d["r_s1"]),
            "S1_Cal_Seen": stats(d["r_s1"], seen_mask),
            "S1_Cal_Unseen": stats(d["r_s1"], ~seen_mask),
            "S1_Cal_Seen_Jgt0": stats(d["r_s1"], seen_mask & (~j0_mask)),
            "S1_Cal_Seen_J0": stats(d["r_s1"], seen_mask & j0_mask),
            "S2_StatusZ_All": stats(d["r_s2"]),
            "S2_StatusZ_Seen": stats(d["r_s2"], seen_mask),
            "S2_StatusZ_Unseen": stats(d["r_s2"], ~seen_mask),
            "S2_RawStatusZ_All": stats(d["r_s2_raw"]),
            "S2_RawStatusZ_Seen": stats(d["r_s2_raw"], seen_mask),
            "S2_RawStatusZ_Unseen": stats(d["r_s2_raw"], ~seen_mask),
            "S2_WithinStatus_Seen": stats(d["r_ws_s1"], seen_mask),
            "S2_WithinStatus_Unseen": stats(d["r_ws_s1"], ~seen_mask),
            "S2_RawWithinStatus_Seen": stats(d["r_ws_s0"], seen_mask),
            "S2_RawWithinStatus_Unseen": stats(d["r_ws_s0"], ~seen_mask),
            "S3_ZeroDirectOverlap_StatusZ": stats(d["r_s2"], j0_mask),
            "S3_ZeroDirectOverlap_RawStatusZ": stats(d["r_s2_raw"], j0_mask),
            "S3_ZeroDirectOverlap_Cal": stats(d["r_s1"], j0_mask),
            "S3_ZeroDirectOverlap_Raw": stats(d["r_s0"], j0_mask),
            "S3_ZeroDirectOverlap_Seen_J0": stats(d["r_s2"], seen_mask & j0_mask),
            "S3_ZeroDirectOverlap_Unseen": stats(d["r_s2"], ~seen_mask),
            "S4_Cumulative_All4": stats(d["r_s4_filt"], s4_mask, equal_disease=True),
            "S4_RawCumulative_NoS1": stats(d["r_s4_raw_filt"], s4_mask, equal_disease=True),
            "S4_Isolated_Dedup_Filt_EqD_on_S1": stats(
                d["r_s1_filt"], nodup_mask, equal_disease=True
            ),
            "S4_Isolated_Dedup_Filt_EqD_on_S0": stats(
                d["r_s0_filt"], nodup_mask, equal_disease=True
            ),
        }

    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"REPORT_V3_SAVED ({len(summary)} models) -> {args.out_json}")
