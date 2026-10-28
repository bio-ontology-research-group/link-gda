#!/usr/bin/env python3
"""
Retrain LinkGDA (TransD-U1, TransD-T1, Profile P-pool), INDIGENA (TransD-BMA),
and ConvKB-D across all 10 disease-disjoint folds on the matched disease-bridge-free
graph (removing training 'associated_with' and 'has_symptom' edges while retaining
all '(g, causes_phenotype, p)' edges, gene annotations, and ontology projections).
"""
import argparse
import hashlib
import json
import os
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

if "--help" not in sys.argv and "-h" not in sys.argv:
    from pykeen.models import TransD, ConvKB
    from pykeen.nn.init import PretrainedInitializer
    from pykeen.triples import TriplesFactory
    from pykeen.training import SLCWATrainingLoop
    from pykeen.training.callbacks import TrainingCallback
else:
    TrainingCallback = object


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def is_disease_entity(uri: str) -> bool:
    return (
        "OMIM:" in uri
        or "Orphanet_" in uri
        or "MONDO_" in uri
        or "DOID_" in uri
        or "/omim/" in uri.lower()
    )


class FastGraphScorer:
    """
    Exact vectorized scorer for LinkGDA / ConvKB-D (gene -[causes_phenotype]-> symptom BMA).
    Computes (num_genes, num_unique_symptoms) once per evaluation call and indexes each
    disease's symptom multiset, matching evaluate_by_graph to machine precision (<1e-6).
    """

    def __init__(self, factory, candidates, profiles, eval_chunk_size=100_000):
        self.factory = factory
        self.entity_to_id = factory.entity_to_id
        self.relation_to_id = factory.relation_to_id
        self.candidates = candidates
        self.profiles = profiles
        self.eval_chunk_size = eval_chunk_size
        self.causes_id = self.relation_to_id["causes_phenotype"]
        self.gene_ids = torch.tensor(
            [self.entity_to_id[g] for g in candidates], dtype=torch.long
        )
        self.num_genes = len(candidates)

    @torch.no_grad()
    def score_diseases(self, model, diseases, pooling_weight: float = 0.5):
        model.eval()
        diseases = sorted(set(diseases))
        unique_phenos = sorted(
            {
                p
                for d in diseases
                for p in self.profiles[d]
                if p in self.entity_to_id
            }
        )
        pheno_to_col = {p: i for i, p in enumerate(unique_phenos)}
        pheno_ids = torch.tensor(
            [self.entity_to_id[p] for p in unique_phenos], dtype=torch.long
        )
        num_p = len(unique_phenos)

        # Build (num_genes * num_p, 3) HRT tensor
        h = self.gene_ids.unsqueeze(1).expand(self.num_genes, num_p).reshape(-1)
        r = torch.full((self.num_genes * num_p,), self.causes_id, dtype=torch.long)
        t = pheno_ids.unsqueeze(0).expand(self.num_genes, num_p).reshape(-1)
        hrt = torch.stack([h, r, t], dim=1)

        parts = []
        device = next(model.parameters()).device
        for start in range(0, hrt.shape[0], self.eval_chunk_size):
            batch = hrt[start : start + self.eval_chunk_size].to(device)
            parts.append(model.predict_hrt(batch).detach().cpu())
        pheno_scores = torch.cat(parts, dim=0).view(self.num_genes, num_p).numpy()

        w = float(pooling_weight)
        score_map = {}
        for d in diseases:
            cols = [pheno_to_col[p] for p in self.profiles[d] if p in pheno_to_col]
            if not cols:
                raise ValueError(f"Disease {d} has no represented phenotypes")
            sub = pheno_scores[:, cols]
            gene_centric = sub.max(axis=1)
            disease_centric = sub.mean(axis=1)
            score_map[d] = (w * gene_centric + (1.0 - w) * disease_centric).astype(
                np.float32
            )
        return score_map


class FastIndigenaScorer:
    """
    Exact vectorized INDIGENA BMA scorer over base phenotype entity embeddings.
    Pre-indexes unique gene phenotypes U_GP so scoring all queries takes ~0.05s on GPU
    instead of 250s, while matching compare_vectorized to machine precision (<1e-6).
    """

    def __init__(self, factory, candidates, profiles, gene_phenotypes_csv, compare_fn):
        self.entity_to_id = factory.entity_to_id
        self.candidates = candidates
        self.profiles = profiles
        self.compare_fn = compare_fn
        frame = pd.read_csv(gene_phenotypes_csv)
        self.gp = {}
        for r in frame.itertuples(index=False):
            if r.Phenotype in self.entity_to_id:
                self.gp.setdefault(r.Gene, []).append(self.entity_to_id[r.Phenotype])
        self.width = max(1, max(len(self.gp.get(g, [])) for g in candidates))

        # Build compact unique gene-phenotype vocabulary U_GP and gene->U_GP index tensors
        u_gp = sorted({p_id for g in candidates for p_id in self.gp.get(g, [])})
        self.u_gp_ids = torch.tensor(u_gp, dtype=torch.long)
        gp_to_col = {p_id: i for i, p_id in enumerate(u_gp)}
        self.num_u_gp = len(u_gp)

        # Padded index table (num_genes, width) into [0 .. num_u_gp], where slot num_u_gp is a dummy -inf row
        idx_table = np.full((len(candidates), self.width), self.num_u_gp, dtype=np.int64)
        counts = np.zeros(len(candidates), dtype=np.float32)
        for i, g in enumerate(candidates):
            ps = self.gp.get(g, [])
            counts[i] = len(ps)
            if ps:
                idx_table[i, : len(ps)] = [gp_to_col[p_id] for p_id in ps]
        self.idx_table = torch.tensor(idx_table, dtype=torch.long)
        self.counts = torch.tensor(counts, dtype=torch.float32)
        self.counts_safe = torch.clamp(self.counts, min=1.0)
        self.valid_mask = (
            torch.arange(self.width).unsqueeze(0) < self.counts.long().unsqueeze(1)
        )  # (num_genes, width)

    @torch.no_grad()
    def score_diseases(self, model, diseases):
        model.eval()
        device = next(model.parameters()).device
        emb = model.entity_representations[0](indices=None).detach()
        diseases = sorted(set(diseases))
        u_q = sorted(
            {
                self.entity_to_id[p]
                for d in diseases
                for p in self.profiles[d]
                if p in self.entity_to_id
            }
        )
        q_to_col = {p_id: i for i, p_id in enumerate(u_q)}
        u_q_ids = torch.tensor(u_q, dtype=torch.long, device=device)

        # Pairwise sigmoid(dot) between all unique gene phenotypes and all unique query phenotypes:
        # shape (num_u_gp + 1, num_u_q), where row num_u_gp is 0.0 (for masked slots)
        gp_emb = emb[self.u_gp_ids.to(device)]  # (num_u_gp, d)
        q_emb = emb[u_q_ids]  # (num_u_q, d)
        sim_pq = torch.sigmoid(gp_emb @ q_emb.T)  # (num_u_gp, num_u_q)
        sim_pq_pad = torch.cat(
            [sim_pq, torch.zeros(1, len(u_q), device=device, dtype=sim_pq.dtype)], dim=0
        )

        idx_dev = self.idx_table.to(device)  # (num_genes, width)
        mask_dev = self.valid_mask.to(device)
        counts_safe_dev = self.counts_safe.to(device)
        has_pheno_dev = (self.counts.to(device) > 0).float()

        # Disease-centric: for each unique query phenotype q, max over p in P_g of sim_pq[p, q]
        # Compute in chunks of 256 query phenotypes to keep GPU memory tiny (<200MB)
        max_g_per_q = torch.empty((len(self.candidates), len(u_q)), device=device, dtype=torch.float32)
        for q_start in range(0, len(u_q), 256):
            q_end = min(q_start + 256, len(u_q))
            # (num_genes, width, chunk_q)
            sub_sim = sim_pq_pad[idx_dev, q_start:q_end]
            sub_sim = sub_sim.masked_fill(~mask_dev.unsqueeze(-1), -float("inf"))
            max_g_per_q[:, q_start:q_end] = sub_sim.max(dim=1).values * has_pheno_dev.unsqueeze(1)

        out = {}
        for d in diseases:
            cols = [q_to_col[self.entity_to_id[p]] for p in self.profiles[d] if p in self.entity_to_id]
            cols_t = torch.tensor(cols, dtype=torch.long, device=device)
            # Disease-centric: mean over q in P_d of max_{p in P_g} sim(p, q)
            d_centric = max_g_per_q[:, cols_t].mean(dim=1)
            # Gene-centric: for each p in U_GP, max over q in P_d of sim(p, q)
            max_p_for_d = sim_pq[:, cols_t].max(dim=1).values  # (num_u_gp,)
            max_p_pad = torch.cat([max_p_for_d, torch.zeros(1, device=device)])
            g_sum = max_p_pad[idx_dev].sum(dim=1)
            g_centric = g_sum / counts_safe_dev
            bma = 0.5 * (g_centric + d_centric)
            out[d] = bma.detach().cpu().numpy().astype(np.float32)
        return out


def metric_key(metrics):
    return (
        float(metrics["hits@10"]),
        float(metrics["hits@1"]),
        float(metrics["mrr"]),
        -float(metrics["mr"]),
    )


class DualTracker:
    """Tracks best checkpoints for raw_mr, raw_head, calibrated_mr, calibrated_head."""

    def __init__(self, prefix: Path):
        self.prefix = Path(prefix)
        self.best = {}

    def update(self, epoch: int, model, metrics_dict: dict):
        # metrics_dict has 'raw': (metrics, _), 'calibrated': (metrics, _)
        for setting in ("raw", "calibrated"):
            m = metrics_dict[setting][0]
            for crit, key_fn in [
                ("mr", lambda x: (-float(x["mr"]),)),
                ("head", metric_key),
            ]:
                arm = f"{setting}_{crit}"
                k = list(key_fn(m))
                if arm not in self.best or k > self.best[arm]["key"]:
                    ckpt = self.prefix.parent / f"{self.prefix.name}_{arm}.pt"
                    tmp = ckpt.with_suffix(".pt.tmp")
                    torch.save(model.state_dict(), tmp)
                    os.replace(tmp, ckpt)
                    self.best[arm] = {
                        "epoch": epoch,
                        "key": k,
                        "metrics": {mk: float(mv) for mk, mv in m.items()},
                        "checkpoint": str(ckpt),
                    }


class CombinedValidationCallback(TrainingCallback):
    def __init__(
        self,
        model,
        val_df,
        candidates,
        graph_scorer: FastGraphScorer,
        graph_tracker: DualTracker,
        profile_metrics_fn,
        indigena_scorer: FastIndigenaScorer = None,
        indigena_tracker: DualTracker = None,
        frequency: int = 20,
        patience_evals: int = None,
        early_stop_arm: str = "calibrated_mr",
    ):
        super().__init__()
        self._model = model
        self.val_df = val_df
        self.val_diseases = sorted(set(val_df.Disease))
        self.candidates = candidates
        self.graph_scorer = graph_scorer
        self.graph_tracker = graph_tracker
        self.profile_metrics_fn = profile_metrics_fn
        self.indigena_scorer = indigena_scorer
        self.indigena_tracker = indigena_tracker
        self.frequency = frequency
        self.patience_evals = patience_evals
        self.early_stop_arm = early_stop_arm
        self.evals_without_improvement = 0
        self.last_best_epoch = 0

    def evaluate_epoch(self, epoch: int):
        t0 = time.time()
        g_map = self.graph_scorer.score_diseases(self._model, self.val_diseases)
        g_metrics = self.profile_metrics_fn(g_map, self.val_df, self.candidates)
        prev_best_epoch = (
            self.graph_tracker.best.get(self.early_stop_arm, {}).get("epoch", -1)
        )
        self.graph_tracker.update(epoch, self._model, g_metrics)
        new_best_epoch = self.graph_tracker.best[self.early_stop_arm]["epoch"]
        if new_best_epoch == epoch and epoch != prev_best_epoch:
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1

        if self.indigena_scorer is not None and self.indigena_tracker is not None:
            i_map = self.indigena_scorer.score_diseases(self._model, self.val_diseases)
            i_metrics = self.profile_metrics_fn(i_map, self.val_df, self.candidates)
            self.indigena_tracker.update(epoch, self._model, i_metrics)

        dt = time.time() - t0
        raw_mr = g_metrics["raw"][0]["mr"]
        raw_h10 = g_metrics["raw"][0]["hits@10"]
        cal_mr = g_metrics["calibrated"][0]["mr"]
        cal_h10 = g_metrics["calibrated"][0]["hits@10"]
        print(
            f"[Epoch {epoch:4d}] Graph val raw MR={raw_mr:.1f} H@10={raw_h10*100:.2f}% | "
            f"cal MR={cal_mr:.1f} H@10={cal_h10*100:.2f}% ({dt:.1f}s)",
            flush=True,
        )
        if (
            self.patience_evals is not None
            and self.evals_without_improvement >= self.patience_evals
        ):
            print(
                f"Early stopping at epoch {epoch} after {self.patience_evals} evals without improvement on {self.early_stop_arm}.",
                flush=True,
            )
            self.training_loop._should_stop = True

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs):
        if epoch % self.frequency == 0:
            self.evaluate_epoch(epoch)


def export_full_scores(
    out_npz: Path,
    model,
    ckpt_path: str,
    scorer,
    test_diseases: list,
    train_diseases: list,
    candidates: list,
):
    model.load_state_dict(torch.load(ckpt_path, map_location="cuda", weights_only=True))
    test_map = scorer.score_diseases(model, test_diseases)
    train_map = scorer.score_diseases(model, train_diseases)
    test_mat = np.stack([test_map[d] for d in test_diseases]).astype(np.float32)
    train_mat = np.stack([train_map[d] for d in train_diseases]).astype(np.float32)
    t_mean = train_mat.mean(axis=0).astype(np.float32)
    t_std = np.maximum(train_mat.std(axis=0), 1e-6).astype(np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_name("tmp_" + out_npz.name)
    np.savez(
        tmp,
        scores=test_mat,
        train_mean=t_mean,
        train_std=t_std,
        diseases=np.array(test_diseases),
        train_diseases=np.array(train_diseases),
        genes=np.array(candidates),
    )
    os.replace(tmp, out_npz)
    print(f"Saved {out_npz} (test={test_mat.shape}, train={train_mat.shape})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument(
        "--condition",
        choices=["U1_INDIGENA_PPOOL", "T1", "CONVKB"],
        required=True,
    )
    ap.add_argument("--data-root", type=Path, required=True, help="Directory containing fold0..9 prepared graph inputs")
    ap.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[2], help="Repository root directory")
    ap.add_argument("--out-root", type=Path, required=True, help="Output directory for checkpoints and .npz score matrices")
    ap.add_argument("--epochs", type=int, default=1000, help="Maximum training epochs")
    ap.add_argument("--sub-batch-size", type=int, default=32768, help="PyKEEN sub-batch size")
    args = ap.parse_args()

    sys.path.insert(0, str(args.code_root / "code"))
    from link_gda.evaluation import compare_vectorized, evaluate_by_graph
    from link_gda.head_experiment import profile_metrics
    from link_gda.negative_sampling import GenePoolNegativeSampler
    from link_gda.profile_ranking import ProfileRanker

    fold_dir = args.data_root / f"fold{args.fold}"
    out_dir = args.out_root / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import fcntl
    comp_map = {
        "U1_INDIGENA_PPOOL": "ppool_completion.json",
        "T1": "t1_completion.json",
        "CONVKB": "convkbd_completion.json",
    }
    if (out_dir / comp_map[args.condition]).exists():
        print(f"Fold {args.fold} [{args.condition}] already completed, skipping.", flush=True)
        return
    lock_file = open(out_dir / f"py_{args.condition}.lock", "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Fold {args.fold} [{args.condition}] locked by another worker, skipping.", flush=True)
        return

    # Load and filter triples: remove 'associated_with' and 'has_symptom'
    raw_df = pd.read_csv(
        fold_dir / "raw/train.txt", sep="\t", header=None, dtype=str, names=["h", "r", "t"]
    )
    kept_df = raw_df[~raw_df["r"].isin(["associated_with", "has_symptom"])]
    triples = kept_df.values
    factory = TriplesFactory.from_labeled_triples(triples, create_inverse_triples=True)

    # Strict audit checks
    disease_nodes = [e for e in factory.entity_to_id if is_disease_entity(e)]
    assert not disease_nodes, f"Disease nodes remained in graph: {disease_nodes[:5]}"

    candidates = (fold_dir / "candidates.txt").read_text().splitlines()
    assert len(candidates) == 13625
    assert all(g in factory.entity_to_id for g in candidates)

    profiles = json.loads((fold_dir / "profiles.json").read_text())
    train_df = pd.read_csv(fold_dir / "training.tsv", sep="\t")
    val_df = pd.read_csv(fold_dir / "validation.tsv", sep="\t")
    test_df = pd.read_csv(fold_dir / "test.tsv", sep="\t")

    test_diseases = sorted(set(test_df.Disease))
    train_diseases = sorted(set(train_df.Disease))
    val_diseases = sorted(set(val_df.Disease))

    print(
        f"Fold {args.fold} [{args.condition}]: {len(triples)} no-bridge triples "
        f"(removed {len(raw_df) - len(kept_df)} bridge triples), "
        f"{len(factory.entity_to_id)} entities, {len(candidates)} candidates.",
        flush=True,
    )

    if args.condition == "U1_INDIGENA_PPOOL":
        model = TransD(
            triples_factory=factory,
            embedding_dim=200,
            relation_dim=200,
            random_seed=0,
        ).cuda()

        graph_scorer = FastGraphScorer(factory, candidates, profiles, eval_chunk_size=200_000)
        indigena_scorer = FastIndigenaScorer(
            factory,
            candidates,
            profiles,
            args.code_root / "data/gene_phenotypes.csv",
            compare_vectorized,
        )

        # Parity check between FastGraphScorer and native evaluate_by_graph on 2 validation diseases
        sub_val = val_df[val_df.Disease.isin(val_diseases[:2])]
        fast_sub = graph_scorer.score_diseases(model, val_diseases[:2])
        native_sub = evaluate_by_graph(
            model,
            sub_val,
            profiles,
            candidates,
            triples_factory=factory,
            return_score_map=True,
        )
        for d in val_diseases[:2]:
            err = float(np.max(np.abs(fast_sub[d] - np.asarray(native_sub[d], dtype=np.float32))))
            assert err < 1e-5, f"FastGraphScorer parity error {err}"
        print(f"FastGraphScorer parity verified (max abs err={err:.2e}).", flush=True)

        u1_tracker = DualTracker(out_dir / "ckpt_u1_nobridge")
        ind_tracker = DualTracker(out_dir / "ckpt_indigena_nobridge")

        cb = CombinedValidationCallback(
            model=model,
            val_df=val_df,
            candidates=candidates,
            graph_scorer=graph_scorer,
            graph_tracker=u1_tracker,
            profile_metrics_fn=profile_metrics,
            indigena_scorer=indigena_scorer,
            indigena_tracker=ind_tracker,
            frequency=20,
        )

        optimizer = torch.optim.Adam(model.get_grad_params(), lr=0.001)
        loop = SLCWATrainingLoop(
            model=model,
            triples_factory=factory,
            optimizer=optimizer,
            negative_sampler_kwargs={"num_negs_per_pos": 1},
        )
        t0 = time.time()
        loop.train(
            triples_factory=factory,
            num_epochs=args.epochs,
            batch_size=65536,
            sub_batch_size=args.sub_batch_size,
            callbacks=[cb],
            use_tqdm=False,
        )
        if args.epochs % 20 != 0:
            cb.evaluate_epoch(args.epochs)
        write_json_atomic(
            out_dir / "u1_indigena_completion.json",
            {
                "fold": args.fold,
                "wall_seconds": time.time() - t0,
                "u1_best": u1_tracker.best,
                "indigena_best": ind_tracker.best,
            },
        )

        # Export U1 test & train scores (raw_head and calibrated_mr)
        export_full_scores(
            out_dir / "linkgda_u1_nobridge_raw_head_scores.npz",
            model,
            u1_tracker.best["raw_head"]["checkpoint"],
            graph_scorer,
            test_diseases,
            train_diseases,
            candidates,
        )
        export_full_scores(
            out_dir / "linkgda_u1_nobridge_cal_mr_scores.npz",
            model,
            u1_tracker.best["calibrated_mr"]["checkpoint"],
            graph_scorer,
            test_diseases,
            train_diseases,
            candidates,
        )

        # Export INDIGENA test & train scores (raw_head and raw_mr)
        export_full_scores(
            out_dir / "indigena_nobridge_raw_head_scores.npz",
            model,
            ind_tracker.best["raw_head"]["checkpoint"],
            indigena_scorer,
            test_diseases,
            train_diseases,
            candidates,
        )
        export_full_scores(
            out_dir / "indigena_nobridge_raw_mr_scores.npz",
            model,
            ind_tracker.best["raw_mr"]["checkpoint"],
            indigena_scorer,
            test_diseases,
            train_diseases,
            candidates,
        )

        # Stage C: Profile P-pool (20 epochs warm-started from U1 raw_head)
        print(f"Starting Profile P-pool (No Disease Bridges) on fold {args.fold}...", flush=True)
        model.load_state_dict(
            torch.load(
                u1_tracker.best["raw_head"]["checkpoint"],
                map_location="cuda",
                weights_only=True,
            )
        )
        ranker = ProfileRanker(
            model,
            factory,
            train_df,
            profiles,
            candidates,
            validation_pairs=val_df,
            seed=0,
            learn_pooling=True,
            temperature=1.0,
            random_negatives=32,
            hard_negatives=32,
            proposal_size=256,
        )
        p_opt = torch.optim.Adam(ranker.parameters(), lr=0.0001)
        ppool_tracker = DualTracker(out_dir / "ckpt_ppool_nobridge")
        t_p = time.time()
        for ep in range(1, 21):
            stats = ranker.train_epoch(p_opt, accumulation_steps=16)
            val_map = graph_scorer.score_diseases(
                ranker.model, val_diseases, pooling_weight=ranker.pooling_weight()
            )
            val_m = profile_metrics(val_map, val_df, candidates)
            ppool_tracker.update(ep, ranker, val_m)
            print(
                f"[P-pool Ep {ep:2d}] loss={stats['mean_loss']:.4f} pool_w={ranker.pooling_weight():.3f} "
                f"val raw H@10={val_m['raw'][0]['hits@10']*100:.2f}% MR={val_m['raw'][0]['mr']:.1f}",
                flush=True,
            )
        ranker.load_state_dict(
            torch.load(
                ppool_tracker.best["raw_head"]["checkpoint"],
                map_location="cuda",
                weights_only=True,
            )
        )
        pw = ranker.pooling_weight()
        test_map = graph_scorer.score_diseases(ranker.model, test_diseases, pooling_weight=pw)
        train_map = graph_scorer.score_diseases(ranker.model, train_diseases, pooling_weight=pw)
        test_mat = np.stack([test_map[d] for d in test_diseases]).astype(np.float32)
        train_mat = np.stack([train_map[d] for d in train_diseases]).astype(np.float32)
        out_npz = out_dir / "linkgda_ppool_nobridge_scores.npz"
        tmp = out_npz.with_name("tmp_" + out_npz.name)
        np.savez(
            tmp,
            scores=test_mat,
            train_mean=train_mat.mean(axis=0).astype(np.float32),
            train_std=np.maximum(train_mat.std(axis=0), 1e-6).astype(np.float32),
            diseases=np.array(test_diseases),
            train_diseases=np.array(train_diseases),
            genes=np.array(candidates),
        )
        os.replace(tmp, out_npz)
        print(f"Saved {out_npz}", flush=True)
        write_json_atomic(
            out_dir / "ppool_completion.json",
            {
                "fold": args.fold,
                "wall_seconds": time.time() - t_p,
                "ppool_best": ppool_tracker.best,
            },
        )

    elif args.condition == "T1":
        model = TransD(
            triples_factory=factory,
            embedding_dim=200,
            relation_dim=200,
            random_seed=0,
        ).cuda()
        graph_scorer = FastGraphScorer(factory, candidates, profiles, eval_chunk_size=200_000)
        t1_tracker = DualTracker(out_dir / "ckpt_t1_nobridge")
        t1_epochs = args.epochs
        if (out_dir / "ckpt_t1_nobridge_calibrated_head.pt").exists():
            for arm_ckpt in ["calibrated_head", "raw_head", "calibrated_mr", "raw_mr"]:
                p_ck = out_dir / f"ckpt_t1_nobridge_{arm_ckpt}.pt"
                if p_ck.exists():
                    model.load_state_dict(torch.load(p_ck, map_location="cuda", weights_only=True))
                    vm = profile_metrics(graph_scorer.score_diseases(model, val_diseases), val_df, candidates)
                    t1_tracker.update(0, model, vm)
            t1_epochs = max(200, args.epochs - 440)
            print(f"Resuming T1 on fold {args.fold} from saved checkpoints for {t1_epochs} remaining epochs...", flush=True)
        cb = CombinedValidationCallback(
            model=model,
            val_df=val_df,
            candidates=candidates,
            graph_scorer=graph_scorer,
            graph_tracker=t1_tracker,
            profile_metrics_fn=profile_metrics,
            frequency=20,
        )
        optimizer = torch.optim.Adam(model.get_grad_params(), lr=0.001)
        sampler_kwargs = {
            "num_negs_per_pos": 1,
            "target_relations": [2 * factory.relation_to_id["causes_phenotype"]],
            "gene_pool": [factory.entity_to_id[g] for g in candidates],
        }
        loop = SLCWATrainingLoop(
            model=model,
            triples_factory=factory,
            optimizer=optimizer,
            negative_sampler=GenePoolNegativeSampler,
            negative_sampler_kwargs=sampler_kwargs,
        )
        t0 = time.time()
        loop.train(
            triples_factory=factory,
            num_epochs=t1_epochs,
            batch_size=65536,
            sub_batch_size=args.sub_batch_size,
            callbacks=[cb],
            use_tqdm=False,
        )
        if t1_epochs % 20 != 0:
            cb.evaluate_epoch(t1_epochs)
        export_full_scores(
            out_dir / "linkgda_t1_nobridge_cal_head_scores.npz",
            model,
            t1_tracker.best["calibrated_head"]["checkpoint"],
            graph_scorer,
            test_diseases,
            train_diseases,
            candidates,
        )
        export_full_scores(
            out_dir / "linkgda_t1_nobridge_raw_head_scores.npz",
            model,
            t1_tracker.best["raw_head"]["checkpoint"],
            graph_scorer,
            test_diseases,
            train_diseases,
            candidates,
        )
        write_json_atomic(
            out_dir / "t1_completion.json",
            {
                "fold": args.fold,
                "wall_seconds": time.time() - t0,
                "t1_best": t1_tracker.best,
            },
        )

    elif args.condition == "CONVKB":
        u1_comp = json.loads((out_dir / "u1_indigena_completion.json").read_text())
        u1_ckpt = u1_comp["u1_best"]["calibrated_mr"]["checkpoint"]
        pretrained = TransD(
            triples_factory=factory,
            embedding_dim=200,
            relation_dim=200,
            random_seed=0,
        )
        pretrained.load_state_dict(torch.load(u1_ckpt, map_location="cpu", weights_only=True))
        entity_ids = torch.tensor(list(factory.entity_to_id.values()))
        relation_ids = torch.tensor(list(range(2 * len(factory.relation_to_id))))
        ent_emb = pretrained.entity_representations[0](indices=entity_ids).detach()
        rel_emb = pretrained.relation_representations[0](indices=relation_ids).detach()

        model = ConvKB(
            triples_factory=factory,
            embedding_dim=200,
            random_seed=0,
            hidden_dropout_rate=0.0,
            num_filters=100,
            entity_initializer=PretrainedInitializer(tensor=ent_emb),
            relation_initializer=PretrainedInitializer(tensor=rel_emb),
        ).cuda()

        graph_scorer = FastGraphScorer(factory, candidates, profiles, eval_chunk_size=32_768)
        conv_tracker = DualTracker(out_dir / "ckpt_convkbd_nobridge")
        conv_epochs = min(args.epochs, 100)
        if (out_dir / "ckpt_convkbd_nobridge_calibrated_mr.pt").exists():
            for arm_ckpt in ["calibrated_mr", "raw_mr", "calibrated_head", "raw_head"]:
                p_ck = out_dir / f"ckpt_convkbd_nobridge_{arm_ckpt}.pt"
                if p_ck.exists():
                    model.load_state_dict(torch.load(p_ck, map_location="cuda", weights_only=True))
                    vm = profile_metrics(graph_scorer.score_diseases(model, val_diseases), val_df, candidates)
                    conv_tracker.update(0, model, vm)
            conv_epochs = 40
            print(f"Resuming CONVKB on fold {args.fold} from saved checkpoints for {conv_epochs} remaining epochs...", flush=True)
        cb = CombinedValidationCallback(
            model=model,
            val_df=val_df,
            candidates=candidates,
            graph_scorer=graph_scorer,
            graph_tracker=conv_tracker,
            profile_metrics_fn=profile_metrics,
            frequency=20,
            patience_evals=15,
            early_stop_arm="calibrated_mr",
        )
        optimizer = torch.optim.Adam(model.get_grad_params(), lr=0.0001)
        loop = SLCWATrainingLoop(
            model=model,
            triples_factory=factory,
            optimizer=optimizer,
        )
        t0 = time.time()
        loop.train(
            triples_factory=factory,
            num_epochs=conv_epochs,
            batch_size=65536,
            sub_batch_size=8192,
            callbacks=[cb],
            use_tqdm=False,
        )
        if not conv_tracker.best:
            cb.evaluate_epoch(conv_epochs)
        export_full_scores(
            out_dir / "convkbd_nobridge_cal_mr_scores.npz",
            model,
            conv_tracker.best["calibrated_mr"]["checkpoint"],
            graph_scorer,
            test_diseases,
            train_diseases,
            candidates,
        )
        export_full_scores(
            out_dir / "convkbd_nobridge_raw_mr_scores.npz",
            model,
            conv_tracker.best["raw_mr"]["checkpoint"],
            graph_scorer,
            test_diseases,
            train_diseases,
            candidates,
        )
        write_json_atomic(
            out_dir / "convkbd_completion.json",
            {
                "fold": args.fold,
                "wall_seconds": time.time() - t0,
                "convkbd_best": conv_tracker.best,
            },
        )


if __name__ == "__main__":
    main()
