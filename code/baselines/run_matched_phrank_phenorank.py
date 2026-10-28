#!/usr/bin/env python3
"""Fold-specific UPheno adaptations of Phrank and PhenoRank.

Human-only and human+MGI Phrank arms use the UPheno taxonomy; native HPO
Phrank is evaluated separately. PhenoRank uses 20 propagation iterations,
rank-based empirical calibration, and the author's decreasing co-occurrence-
tier sampler with uncapped query lengths. --sampler legacy-weighted reproduces
the earlier weighted-neighbour, length-20 adaptation. Both the ontology/source
adaptations and balanced-source variants are explicitly recorded in provenance.
"""
import concurrent.futures
import json
import math
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from link_gda.phenorank_simulation import cooccurrence_tiers, simulate_disease
from link_gda.score_io import sha256
import numpy as np
from scipy import sparse
from scipy.stats import rankdata

import csv

BASE_DIR = Path("workspace")
DATA_DIR = BASE_DIR / "data"
OUT_ROOT = BASE_DIR / "baselines" / "refitted"
PHENORANK_RES_DIR = BASE_DIR / "baselines" / "phenorank" / "resources"
PHENORANK_ORIG_DIR = PHENORANK_RES_DIR / "original"
SAMPLER = "published-tier"
SIMULATIONS = 1000

PHENO_PREFIXES = (
    "http://purl.obolibrary.org/obo/HP_",
    "http://purl.obolibrary.org/obo/MP_",
    "http://purl.obolibrary.org/obo/UPHENO_",
    "http://purl.obolibrary.org/obo/ZP_",
    "http://purl.obolibrary.org/obo/WBPhenotype_",
    "http://purl.obolibrary.org/obo/FYPO_",
    "http://purl.obolibrary.org/obo/FBcv_",
    "http://purl.obolibrary.org/obo/APO_",
    "http://purl.obolibrary.org/obo/DDPHENO_",
    "http://purl.obolibrary.org/obo/TO_",
    "http://purl.obolibrary.org/obo/PHIPO_",
    "http://purl.obolibrary.org/obo/PLANP_",
    "http://purl.obolibrary.org/obo/XPO_",
    "http://purl.obolibrary.org/obo/MGPO_",
)


def is_pheno(uri: str) -> bool:
    return uri.startswith(PHENO_PREFIXES)


def load_pairs(tsv_path: Path):
    pairs = []
    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        g_idx = header.index("Gene")
        d_idx = header.index("Disease")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            pairs.append((parts[g_idx], parts[d_idx]))
    return pairs


def load_ppi_rwr_matrix(candidates: list):
    """
    Load PhenoRank's human PPI network (W_generic.pickle and phenorank_genes.tsv),
    map our 13,625 candidate genes via hgnc_complete_set.txt exactly as in
    `baselines/phenorank/run.py`, and precompute the 20-step RWR operator restricted
    to our 13,625 candidates:
      S^{(t+1)} = (1 - r) * W * S^{(t)} + r * S^{(0)}
    Since initial scores S^{(0)} are supported on candidate genes (or all PPI genes),
    precomputing the 13,625 x 13,625 effective 20-step RWR matrix M_rwr by propagating
    from the full 19,200-gene W_generic network makes every subsequent RWR propagation
    a single fast matrix multiplication `S0 @ M_rwr.T`!
    """
    pr_genes = (PHENORANK_ORIG_DIR / "phenorank_genes.tsv").read_text().splitlines()
    reverse = defaultdict(set)
    with (PHENORANK_RES_DIR / "hgnc_complete_set.txt").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["status"] == "Approved" and r["entrez_id"] and r["ensembl_gene_id"]:
                reverse[r["entrez_id"]].add(r["ensembl_gene_id"])
    idmap = {}
    for g in candidates:
        hits = reverse.get(g.rsplit("/", 1)[-1], set())
        idmap[g] = next(iter(hits)) if len(hits) == 1 else "UNMAPPED:" + g

    original_count = len(pr_genes)
    all_genes = pr_genes + sorted(set(idmap.values()) - set(pr_genes))
    gi = {g: i for i, g in enumerate(all_genes)}
    outix = np.array([gi[idmap[g]] for g in candidates], dtype=np.int64)

    with (PHENORANK_ORIG_DIR / "W_generic.pickle").open("rb") as f:
        W0 = pickle.load(f, encoding="latin1").tocsr()
    W_full = sparse.block_diag(
        (W0, sparse.csr_matrix((len(all_genes) - original_count,) * 2)), format="csr"
    )
    return W_full.astype(np.float32), outix, len(all_genes)


def rwr_propagate(
    W_tuple, S0_cand: np.ndarray, r: float = 0.1, steps: int = 20
):
    """
    Exact 20-iteration Random Walk with Restart on the full PPI network W_full,
    scattering candidate initial scores S0_cand (N_queries, 13625) into the full
    gene space (N_queries, N_all_genes), propagating 20 steps with W_full.T, and
    gathering [:, outix] back to (N_queries, 13625).
    """
    W_full, outix, n_all = W_tuple
    WT = W_full.T.tocsr()
    n_q = S0_cand.shape[0]
    S_full = np.zeros((n_q, n_all), dtype=np.float32)
    S_full[:, outix] = S0_cand.astype(np.float32)
    r_S0 = (r * S_full).astype(np.float32)
    one_minus_r = np.float32(1.0 - r)
    for _ in range(steps):
        S_full = one_minus_r * (S_full @ WT) + r_S0
    return S_full[:, outix].copy()


def save_npz(out_path: Path, scores: np.ndarray, train_scores: np.ndarray, test_diseases: list, train_diseases: list, candidates: list):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name("tmp_" + out_path.name)
    if train_scores is not None and train_scores.ndim == 2:
        t_mean = train_scores.mean(axis=0).astype(np.float32)
        t_std = np.maximum(train_scores.std(axis=0), 1e-6).astype(np.float32)
    else:
        t_mean = np.zeros(len(candidates), dtype=np.float32)
        t_std = np.ones(len(candidates), dtype=np.float32)
    np.savez(
        tmp,
        scores=scores.astype(np.float32),
        train_mean=t_mean,
        train_std=t_std,
        diseases=np.array(test_diseases),
        train_diseases=np.array(train_diseases),
        genes=np.array(candidates),
    )
    os.replace(tmp, out_path)


def compute_phrank_mic(profile_closures: list, all_terms: list, term_to_idx: dict, parents_map: dict):
    """
    Compute Phrank Marginal Information Content (MIC) over a collection of gene closures:
      MIC(t) = -log2( |G(t)| / |U_{pa in Parents(t)} G(pa)| )
    Root terms (where parent union is empty) have MIC(t) = 0.
    """
    term_genes = defaultdict(set)
    for g_idx, closure in enumerate(profile_closures):
        for t in closure:
            term_genes[t].add(g_idx)

    mic = np.zeros(len(all_terms), dtype=np.float32)
    for t, t_idx in term_to_idx.items():
        g_t = term_genes.get(t)
        if not g_t:
            continue
        pars = parents_map.get(t, ())
        if not pars:
            continue
        par_genes = set()
        for p in pars:
            if p in term_genes:
                par_genes.update(term_genes[p])
        if len(par_genes) > len(g_t) > 0:
            mic[t_idx] = -math.log2(len(g_t) / len(par_genes))
    return mic


def build_binary_csr(closures: list, term_to_idx: dict, n_terms: int):
    rows = []
    cols = []
    for r_idx, cl in enumerate(closures):
        for t in cl:
            if t in term_to_idx:
                rows.append(r_idx)
                cols.append(term_to_idx[t])
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(closures), n_terms), dtype=np.float32)


def process_fold(fold: int):
    t0 = time.time()
    fold_dir = DATA_DIR / f"fold{fold}"
    out_dir = OUT_ROOT / f"fold{fold}"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'Use a fresh output directory for this refit: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = (fold_dir / "candidates.txt").read_text().splitlines()
    cand_to_idx = {g: i for i, g in enumerate(candidates)}
    n_genes = len(candidates)

    profiles = json.loads((fold_dir / "profiles.json").read_text())
    train_pairs = load_pairs(fold_dir / "training.tsv")
    test_pairs = load_pairs(fold_dir / "test.tsv")

    train_diseases = sorted({d for _, d in train_pairs})
    test_diseases = sorted({d for _, d in test_pairs})
    train_d_to_idx = {d: i for i, d in enumerate(train_diseases)}

    # Parse raw/train.txt for UPheno subclass edges and MGI mouse has_phenotype edges
    parents_map = defaultdict(set)
    mouse_direct = defaultdict(set)
    with open(fold_dir / "raw" / "train.txt") as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            if r == "has_phenotype" and h in cand_to_idx:
                mouse_direct[h].add(t)
            elif r == "http://subclassof" and is_pheno(h) and is_pheno(t) and h != t:
                parents_map[h].add(t)
            elif r == "http://superclassof" and is_pheno(h) and is_pheno(t) and h != t:
                parents_map[t].add(h)

    # Transitive ancestor closure with memoization
    anc_cache = {}

    def get_ancestors(term):
        if term in anc_cache:
            return anc_cache[term]
        visited = {term}
        stack = [term]
        while stack:
            cur = stack.pop()
            for p in parents_map.get(cur, ()):
                if p not in visited:
                    visited.add(p)
                    stack.append(p)
        anc_cache[term] = visited
        return visited

    def close_profile(terms):
        cl = set()
        for t in terms:
            cl.update(get_ancestors(t))
        return cl

    # Precompute closures for training diseases, test diseases, and mouse gene profiles
    train_d_closures = [close_profile(profiles[d]) for d in train_diseases]
    test_d_closures = [close_profile(profiles[d]) for d in test_diseases]
    mouse_g_closures = [close_profile(mouse_direct.get(g, ())) for g in candidates]

    # Map genes to training disease indices
    gene_train_d_indices = [[] for _ in range(n_genes)]
    for g, d in train_pairs:
        if g in cand_to_idx and d in train_d_to_idx:
            gene_train_d_indices[cand_to_idx[g]].append(train_d_to_idx[d])
    for i in range(n_genes):
        gene_train_d_indices[i] = sorted(set(gene_train_d_indices[i]))

    # Active phenotype vocabulary across all closures
    all_terms = sorted(
        set()
        .union(*train_d_closures)
        .union(*test_d_closures)
        .union(*mouse_g_closures)
    )
    term_to_idx = {t: i for i, t in enumerate(all_terms)}
    n_terms = len(all_terms)

    Q_test = build_binary_csr(test_d_closures, term_to_idx, n_terms)
    Q_train = build_binary_csr(train_d_closures, term_to_idx, n_terms)
    B_train_d = Q_train  # (n_train_d, n_terms)
    B_mouse_g = build_binary_csr(mouse_g_closures, term_to_idx, n_terms)  # (n_genes, n_terms)

    # Human gene union closures & Human+MGI gene union closures
    hpo_g_closures = []
    hm_g_closures = []
    for i in range(n_genes):
        u_h = set()
        for d_idx in gene_train_d_indices[i]:
            u_h.update(train_d_closures[d_idx])
        hpo_g_closures.append(u_h)
        hm_g_closures.append(u_h | mouse_g_closures[i])

    B_hpo_g = build_binary_csr(hpo_g_closures, term_to_idx, n_terms)
    B_hm_g = build_binary_csr(hm_g_closures, term_to_idx, n_terms)

    # =========================================================================
    # PART 1: PHRANK (Jagadeesh et al., 2019) - 10 Matched Variants
    # =========================================================================
    mic_hpo = compute_phrank_mic(hpo_g_closures, all_terms, term_to_idx, parents_map)
    mic_hm = compute_phrank_mic(hm_g_closures, all_terms, term_to_idx, parents_map)
    mic_mgi = compute_phrank_mic(mouse_g_closures, all_terms, term_to_idx, parents_map)

    def score_phrank_disease_max(mic_vec, include_mouse: bool):
        W_mic = sparse.diags(mic_vec)
        # Disease-by-disease Phrank similarity
        # Test vs train_d: (n_test, n_train_d)
        sim_test_d = (Q_test @ W_mic @ B_train_d.T).toarray()
        # Train vs train_d: (n_train, n_train_d) with diagonal masked for LOO
        sim_train_d = (Q_train @ W_mic @ B_train_d.T).toarray()
        np.fill_diagonal(sim_train_d, -1.0)

        self_d = np.maximum(np.asarray(B_train_d @ mic_vec).reshape(-1), 1e-12)
        sim_test_d_norm = sim_test_d / self_d[None, :]
        sim_train_d_norm = sim_train_d / self_d[None, :]
        np.fill_diagonal(sim_train_d_norm, -1.0)

        if include_mouse:
            sim_test_m = (Q_test @ W_mic @ B_mouse_g.T).toarray()
            sim_train_m = (Q_train @ W_mic @ B_mouse_g.T).toarray()
            self_m = np.maximum(np.asarray(B_mouse_g @ mic_vec).reshape(-1), 1e-12)
            sim_test_m_norm = sim_test_m / self_m[None, :]
            sim_train_m_norm = sim_train_m / self_m[None, :]

            test_raw = sim_test_m.copy()
            train_raw = sim_train_m.copy()
            test_norm = sim_test_m_norm.copy()
            train_norm = sim_train_m_norm.copy()
        else:
            test_raw = np.zeros((len(test_diseases), n_genes), dtype=np.float32)
            train_raw = np.zeros((len(train_diseases), n_genes), dtype=np.float32)
            test_norm = np.zeros((len(test_diseases), n_genes), dtype=np.float32)
            train_norm = np.zeros((len(train_diseases), n_genes), dtype=np.float32)

        for g_idx, d_idxs in enumerate(gene_train_d_indices):
            if not d_idxs:
                continue
            d_max_test_raw = sim_test_d[:, d_idxs].max(axis=1)
            d_max_test_norm = sim_test_d_norm[:, d_idxs].max(axis=1)
            # For train queries, if a gene has only 1 training disease d0, its LOO human score on d0 is 0
            sub_train_raw = np.maximum(sim_train_d[:, d_idxs], 0.0)
            sub_train_norm = np.maximum(sim_train_d_norm[:, d_idxs], 0.0)
            d_max_train_raw = sub_train_raw.max(axis=1)
            d_max_train_norm = sub_train_norm.max(axis=1)

            if include_mouse:
                test_raw[:, g_idx] = np.maximum(test_raw[:, g_idx], d_max_test_raw)
                train_raw[:, g_idx] = np.maximum(train_raw[:, g_idx], d_max_train_raw)
                test_norm[:, g_idx] = np.maximum(test_norm[:, g_idx], d_max_test_norm)
                train_norm[:, g_idx] = np.maximum(train_norm[:, g_idx], d_max_train_norm)
            else:
                test_raw[:, g_idx] = d_max_test_raw
                train_raw[:, g_idx] = d_max_train_raw
                test_norm[:, g_idx] = d_max_test_norm
                train_norm[:, g_idx] = d_max_train_norm

        return test_raw, train_raw, test_norm, train_norm

    def score_phrank_direct(mic_vec, B_target, loo_human: bool, include_mouse: bool):
        W_mic = sparse.diags(mic_vec)
        test_raw = (Q_test @ W_mic @ B_target.T).toarray()
        train_raw = (Q_train @ W_mic @ B_target.T).toarray()
        self_g = np.maximum(np.asarray(B_target @ mic_vec).reshape(-1), 1e-12)
        test_norm = test_raw / self_g[None, :]
        train_norm = train_raw / self_g[None, :]

        if loo_human:
            # Recompute LOO direct union score for each training disease d0 and its causal genes
            for g_idx, d_idxs in enumerate(gene_train_d_indices):
                if not d_idxs:
                    continue
                for d0 in d_idxs:
                    rem_cl = set(mouse_g_closures[g_idx]) if include_mouse else set()
                    for d_other in d_idxs:
                        if d_other != d0:
                            rem_cl.update(train_d_closures[d_other])
                    if not rem_cl:
                        train_raw[d0, g_idx] = 0.0
                        train_norm[d0, g_idx] = 0.0
                    else:
                        rem_idxs = [term_to_idx[t] for t in rem_cl if t in term_to_idx]
                        q_cl = train_d_closures[d0]
                        inter_idxs = [term_to_idx[t] for t in (rem_cl & q_cl) if t in term_to_idx]
                        r_score = float(mic_vec[inter_idxs].sum()) if inter_idxs else 0.0
                        s_score = max(float(mic_vec[rem_idxs].sum()), 1e-12)
                        train_raw[d0, g_idx] = r_score
                        train_norm[d0, g_idx] = r_score / s_score

        return test_raw, train_raw, test_norm, train_norm

    # 1-2. Phrank-HPO (disease_max, raw & norm)
    tr_raw, trn_raw, tr_norm, trn_norm = score_phrank_disease_max(mic_hpo, include_mouse=False)
    save_npz(out_dir / "phrank_human_upheno_disease_raw_scores.npz", tr_raw, trn_raw, test_diseases, train_diseases, candidates)
    save_npz(out_dir / "phrank_human_upheno_disease_norm_scores.npz", tr_norm, trn_norm, test_diseases, train_diseases, candidates)

    # 3-4. Phrank-HPO (direct union, raw & norm)
    tr_raw, trn_raw, tr_norm, trn_norm = score_phrank_direct(mic_hpo, B_hpo_g, loo_human=True, include_mouse=False)
    save_npz(out_dir / "phrank_human_upheno_direct_raw_scores.npz", tr_raw, trn_raw, test_diseases, train_diseases, candidates)
    save_npz(out_dir / "phrank_human_upheno_direct_norm_scores.npz", tr_norm, trn_norm, test_diseases, train_diseases, candidates)

    # 5-6. Phrank-UPheno Human+MGI (disease_max, raw & norm)
    tr_raw, trn_raw, tr_norm, trn_norm = score_phrank_disease_max(mic_hm, include_mouse=True)
    save_npz(out_dir / "phrank_upheno_hm_disease_raw_scores.npz", tr_raw, trn_raw, test_diseases, train_diseases, candidates)
    save_npz(out_dir / "phrank_upheno_hm_disease_norm_scores.npz", tr_norm, trn_norm, test_diseases, train_diseases, candidates)

    # 7-8. Phrank-UPheno Human+MGI (direct union, raw & norm)
    tr_raw, trn_raw, tr_norm, trn_norm = score_phrank_direct(mic_hm, B_hm_g, loo_human=True, include_mouse=True)
    save_npz(out_dir / "phrank_upheno_hm_direct_raw_scores.npz", tr_raw, trn_raw, test_diseases, train_diseases, candidates)
    save_npz(out_dir / "phrank_upheno_hm_direct_norm_scores.npz", tr_norm, trn_norm, test_diseases, train_diseases, candidates)

    # 9-10. Phrank-UPheno MGI-Only (raw & norm)
    tr_raw, trn_raw, tr_norm, trn_norm = score_phrank_direct(mic_mgi, B_mouse_g, loo_human=False, include_mouse=True)
    save_npz(out_dir / "phrank_upheno_mgi_raw_scores.npz", tr_raw, trn_raw, test_diseases, train_diseases, candidates)
    save_npz(out_dir / "phrank_upheno_mgi_norm_scores.npz", tr_norm, trn_norm, test_diseases, train_diseases, candidates)

    # =========================================================================
    # PART 2: PHENORANK (Cornish et al., 2018) - Matched UPheno v2 + MGI + PPI
    # =========================================================================
    W_ppi = load_ppi_rwr_matrix(candidates)

    # Term Information Content (Equation E1 of Cornish et al., 2018):
    # (a) Human + Mouse condition matrix C_HM = [B_train_d; B_mouse_g]
    n_cond_hm = B_train_d.shape[0] + B_mouse_g.shape[0]
    counts_hm = np.asarray(B_train_d.sum(axis=0) + B_mouse_g.sum(axis=0)).reshape(-1)
    ic_hm = np.where(counts_hm > 0, -np.log2(np.maximum(counts_hm, 1.0) / n_cond_hm), 0.0).astype(np.float32)

    # (b) Mouse-only condition matrix C_M = B_mouse_g (for include_h=False variants)
    n_cond_m = B_mouse_g.shape[0]
    counts_m = np.asarray(B_mouse_g.sum(axis=0)).reshape(-1)
    ic_m = np.where(counts_m > 0, -np.log2(np.maximum(counts_m, 1.0) / n_cond_m), 0.0).astype(np.float32)

    # Disease -> Gene mapping matrix M_d2g (n_train_d, n_genes)
    d_rows = []
    g_cols = []
    for g_idx, d_idxs in enumerate(gene_train_d_indices):
        for d_idx in d_idxs:
            d_rows.append(d_idx)
            g_cols.append(g_idx)
    M_d2g = sparse.csr_matrix(
        (np.ones(len(d_rows), dtype=np.float32), (d_rows, g_cols)),
        shape=(len(train_diseases), n_genes),
        dtype=np.float32,
    )
    n_omims_h = np.asarray(M_d2g.sum(axis=0)).reshape(-1).astype(np.float32)
    has_h_mask = (n_omims_h > 0).astype(np.float32)
    denom_h = np.maximum(n_omims_h, 1.0)
    denom_sources = 1.0 + has_h_mask  # 2.0 for Seen genes, 1.0 for Unseen genes

    # Pre-weight condition matrices by IC for fast vectorized SimGIC
    B_train_d_ic_hm = B_train_d @ sparse.diags(ic_hm)
    s_train_d_hm = np.asarray(B_train_d @ ic_hm).reshape(-1)

    B_mouse_g_ic_hm = B_mouse_g @ sparse.diags(ic_hm)
    s_mouse_g_hm = np.asarray(B_mouse_g @ ic_hm).reshape(-1)

    B_mouse_g_ic_m = B_mouse_g @ sparse.diags(ic_m)
    s_mouse_g_m = np.asarray(B_mouse_g @ ic_m).reshape(-1)

    def compute_phenorank_continuous(Q_batch: sparse.csr_matrix, is_train_loo: bool = False):
        """
        Compute initial SimGIC gene scores and 20-step RWR propagated scores for:
          - hm_add: Additive Eq. E2 (score_h + score_m) + RWR on W_ppi
          - hm_bal: Balanced species average ((score_h + score_m) / (1 + I(n_omims_h > 0))) + RWR on W_ppi
          - mgi_ppi: Mouse-only (include_h=False) + RWR on W_ppi
          - mgi_noppi: Mouse-only (include_h=False, W=0)
        """
        # 1. Human SimGIC (using ic_hm)
        s_q_hm = np.asarray(Q_batch @ ic_hm).reshape(-1, 1)
        inter_h = (Q_batch @ B_train_d_ic_hm.T).toarray()
        union_h = np.maximum(s_q_hm + s_train_d_hm[None, :] - inter_h, 1e-12)
        sim_h = inter_h / union_h  # (B, n_train_d)
        sim_h = np.where(s_train_d_hm[None, :] > 0, sim_h, 0.0).astype(np.float32)

        if not is_train_loo:
            score_h = (sim_h @ M_d2g) / denom_h[None, :]
        else:
            # LOO: mask diagonal of sim_h (each training query d_0 excludes its own condition d_0)
            np.fill_diagonal(sim_h, 0.0)
            sum_h = sim_h @ M_d2g
            # For (d0, g) in training pairs, decrement denominator by 1
            denom_loo = np.broadcast_to(n_omims_h[None, :], sum_h.shape).copy()
            denom_loo[d_rows, g_cols] -= 1.0
            score_h = np.where(denom_loo > 0, sum_h / np.maximum(denom_loo, 1.0), 0.0).astype(np.float32)

        # 2. Mouse SimGIC (using ic_hm for HM variants)
        inter_m_hm = (Q_batch @ B_mouse_g_ic_hm.T).toarray()
        union_m_hm = np.maximum(s_q_hm + s_mouse_g_hm[None, :] - inter_m_hm, 1e-12)
        score_m_hm = (inter_m_hm / union_m_hm).astype(np.float32)
        score_m_hm = np.where(s_mouse_g_hm[None, :] > 0, score_m_hm, 0.0)

        # 3. Mouse SimGIC (using ic_m for MGI-Only variants, matching include_h=False)
        s_q_m = np.asarray(Q_batch @ ic_m).reshape(-1, 1)
        inter_m_only = (Q_batch @ B_mouse_g_ic_m.T).toarray()
        union_m_only = np.maximum(s_q_m + s_mouse_g_m[None, :] - inter_m_only, 1e-12)
        score_m_only = (inter_m_only / union_m_only).astype(np.float32)
        score_m_only = np.where(s_mouse_g_m[None, :] > 0, score_m_only, 0.0)

        s0_hm_add = score_h + score_m_hm
        if not is_train_loo:
            s0_hm_bal = s0_hm_add / denom_sources[None, :]
        else:
            has_h_loo = (denom_loo > 0).astype(np.float32)
            s0_hm_bal = s0_hm_add / (1.0 + has_h_loo)

        prop_hm_add = rwr_propagate(W_ppi, s0_hm_add)
        prop_hm_bal = rwr_propagate(W_ppi, s0_hm_bal)
        prop_mgi_ppi = rwr_propagate(W_ppi, score_m_only)
        prop_mgi_noppi = score_m_only

        return {
            "hm_add": prop_hm_add,
            "hm_bal": prop_hm_bal,
            "mgi_ppi": prop_mgi_ppi,
            "mgi_noppi": prop_mgi_noppi,
        }

    obs_test = compute_phenorank_continuous(Q_test, is_train_loo=False)
    obs_train = compute_phenorank_continuous(Q_train, is_train_loo=True)

    # Save continuous RWR (`NoSimulation`) variants
    for key in ("hm_add", "hm_bal", "mgi_ppi", "mgi_noppi"):
        save_npz(
            out_dir / f"phenorank_upheno_{key}_nosim_scores.npz",
            obs_test[key],
            obs_train[key],
            test_diseases,
            train_diseases,
            candidates,
        )

    cooccur_counts = defaultdict(lambda: defaultdict(int))
    for d in train_diseases:
        ps = sorted({p for p in profiles[d] if p in term_to_idx})
        for p1 in ps:
            cooccur_counts[p1]  # ensure key exists
            for p2 in ps:
                if p1 != p2:
                    cooccur_counts[p1][p2] += 1

    anchor_terms = sorted(cooccur_counts.keys())
    cooccur_neighbors = {}
    for p1 in anchor_terms:
        nbrs = cooccur_counts[p1]
        if nbrs:
            ks = list(nbrs.keys())
            ws = np.array([nbrs[k] for k in ks], dtype=np.float64)
            ws /= ws.sum()
            cooccur_neighbors[p1] = (ks, ws)
        else:
            cooccur_neighbors[p1] = ([], np.array([]))

    def legacy_simulate_disease_closures(num_phenos: int, n_sims: int, rng: np.random.Generator):
        closures = []
        n_anchors = len(anchor_terms)
        for _ in range(n_sims):
            anchor = anchor_terms[int(rng.integers(n_anchors))]
            chosen = {anchor}
            ks, ws = cooccur_neighbors[anchor]
            if num_phenos > 1 and len(ks) > 0:
                take = min(num_phenos - 1, len(ks))
                sampled = rng.choice(len(ks), size=take, replace=False, p=ws)
                for idx in sampled:
                    chosen.add(ks[int(idx)])
            while len(chosen) < num_phenos:
                extra = anchor_terms[int(rng.integers(n_anchors))]
                chosen.add(extra)
            closures.append(close_profile(chosen))
        return build_binary_csr(closures, term_to_idx, n_terms)


    tiers = cooccurrence_tiers(profiles[d] for d in train_diseases)

    def simulate_disease_closures(num_phenos, n_sims, rng):
        if SAMPLER == 'legacy-weighted':
            return legacy_simulate_disease_closures(num_phenos, n_sims, rng)
        closures = [close_profile(simulate_disease(num_phenos, tiers, rng)) for _ in range(n_sims)]
        return build_binary_csr(closures, term_to_idx, n_terms)

    N_SIMS = SIMULATIONS
    # Published-tier preserves the observed query length. Only the explicitly
    # archived legacy adaptation retains its former length-20 truncation.
    test_lengths = [len(set(profiles[d])) for d in test_diseases]
    if any(length == 0 for length in test_lengths):
        raise ValueError('Empty test phenotype profile')
    if SAMPLER == 'legacy-weighted':
        test_lengths = [min(length, 20) for length in test_lengths]
    unique_lengths = sorted(set(test_lengths))

    # Pre-rank observed scores within each test query (Cornish et al.: `score_obs = ss.rankdata(gene_scores_prop)`)
    obs_test_ranks = {
        k: np.vstack([rankdata(row) for row in obs_test[k]]).astype(np.float32)
        for k in obs_test
    }

    sim_test_strict = {
        k: np.zeros((len(test_diseases), n_genes), dtype=np.float32) for k in obs_test
    }
    sim_test_tb = {
        k: np.zeros((len(test_diseases), n_genes), dtype=np.float32) for k in obs_test
    }

    for m in unique_lengths:
        rng = (np.random.default_rng(fold * 10000 + m) if SAMPLER == 'legacy-weighted'
               else np.random.RandomState(fold * 10000 + m))
        Q_sim = simulate_disease_closures(m, N_SIMS, rng)
        sim_prop = compute_phenorank_continuous(Q_sim, is_train_loo=False)

        test_q_idxs = [i for i, L in enumerate(test_lengths) if L == m]

        for k in ("hm_add", "hm_bal", "mgi_ppi", "mgi_noppi"):
            # Rank within each of the 1,000 simulated queries: shape (1000, n_genes)
            sim_ranks = np.vstack([rankdata(row) for row in sim_prop[k]]).astype(np.float32)
            sim_ranks_sorted = np.sort(sim_ranks, axis=0)  # (1000, n_genes)

            if test_q_idxs:
                obs_r = obs_test_ranks[k][test_q_idxs]  # (n_q, n_genes)
                counts_ge = np.empty_like(obs_r, dtype=np.float32)
                for g_col in range(n_genes):
                    idx_left = np.searchsorted(sim_ranks_sorted[:, g_col], obs_r[:, g_col], side="left")
                    counts_ge[:, g_col] = N_SIMS - idx_left
                p_emp = np.maximum(counts_ge, 1.0) / float(N_SIMS)
                sim_test_strict[k][test_q_idxs] = -p_emp
                sim_test_tb[k][test_q_idxs] = -p_emp + 1e-6 * (obs_r / float(n_genes))

    for k in ("hm_add", "hm_bal", "mgi_ppi", "mgi_noppi"):
        save_npz(
            out_dir / f"phenorank_upheno_{k}_sim_scores.npz",
            sim_test_strict[k],
            None,
            test_diseases,
            train_diseases,
            candidates,
        )
        save_npz(
            out_dir / f"phenorank_upheno_{k}_sim_tb_scores.npz",
            sim_test_tb[k],
            None,
            test_diseases,
            train_diseases,
            candidates,
        )

    provenance = {'fold': fold, 'sampler': SAMPLER, 'simulations': SIMULATIONS,
                  'query_length_cap': 20 if SAMPLER == 'legacy-weighted' else None,
                  'simulation_reference': 'inner-training human disease profiles',
                  'ontology': 'UPheno taxonomy; human-only arms also use this hierarchy',
                  'rwr_iterations': 20, 'restart_probability': 0.1,
                  'empirical_p': 'max(number of null ranks >= observed rank, 1) / simulations',
                  'tie_breaking': '1e-6 times observed ascending rank divided by candidate count',
                  'source_sha256': sha256(__file__),
                  'sampler_source_sha256': sha256(Path(__file__).resolve().parents[1] / 'link_gda/phenorank_simulation.py'),
                  'inputs': {name: sha256(fold_dir / name) for name in
                             ['training.tsv', 'test.tsv', 'profiles.json', 'candidates.txt', 'raw/train.txt']},
                  'outputs': {path.name: sha256(path) for path in sorted(out_dir.glob('*.npz'))}}
    (out_dir / 'COMPLETE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    dt = time.time() - t0
    print(f"Completed fold {fold} in {dt:.1f}s", flush=True)
    return fold, dt


def configure(paths):
    global BASE_DIR, DATA_DIR, OUT_ROOT, PHENORANK_RES_DIR, PHENORANK_ORIG_DIR, SAMPLER, SIMULATIONS
    BASE_DIR = Path(paths['base'])
    DATA_DIR = BASE_DIR / 'data'
    OUT_ROOT = Path(paths['output'])
    PHENORANK_RES_DIR = Path(paths['resources'])
    PHENORANK_ORIG_DIR = PHENORANK_RES_DIR / 'original'
    SAMPLER, SIMULATIONS = paths['sampler'], paths['simulations']


def main():
    global BASE_DIR, DATA_DIR, OUT_ROOT, PHENORANK_RES_DIR, PHENORANK_ORIG_DIR, SAMPLER, SIMULATIONS
    import argparse
    ap = argparse.ArgumentParser(description="Run matched 10-fold UPheno v2 + MGI Phrank and PhenoRank evaluation.")
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR, help="Base workspace directory containing data/ and baselines/")
    ap.add_argument("--out-root", type=Path, default=None, help="Output directory for .npz score matrices")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)), help="Folds to evaluate (default: 0..9)")
    ap.add_argument("--workers", type=int, default=10, help="Number of parallel fold workers")
    ap.add_argument('--sampler', choices=['published-tier', 'legacy-weighted'], default='published-tier')
    ap.add_argument('--simulations', type=int, default=1000)
    ap.add_argument('--phenorank-resources', type=Path)
    args = ap.parse_args()
    if args.simulations < 1 or args.workers < 1:
        ap.error('Simulations and workers must be positive')
    SAMPLER, SIMULATIONS = args.sampler, args.simulations

    BASE_DIR = args.base_dir
    DATA_DIR = BASE_DIR / "data"
    OUT_ROOT = args.out_root if args.out_root is not None else (BASE_DIR / "baselines" / "refitted")
    PHENORANK_RES_DIR = args.phenorank_resources or BASE_DIR / "baselines" / "phenorank" / "resources"
    PHENORANK_ORIG_DIR = PHENORANK_RES_DIR / "original"

    t_start = time.time()
    print("Running matched 10-fold Phrank and PhenoRank evaluation...", flush=True)
    paths = dict(base=str(BASE_DIR), output=str(OUT_ROOT), resources=str(PHENORANK_RES_DIR),
                 sampler=SAMPLER, simulations=SIMULATIONS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=configure, initargs=(paths,)) as ex:
        futs = [ex.submit(process_fold, f) for f in args.folds]
        for fut in concurrent.futures.as_completed(futs):
            fold, dt = fut.result()
    print(f"All {len(args.folds)} folds completed in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
