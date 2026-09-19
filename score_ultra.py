"""Zero-shot ULTRA scoring driver for the LinkGDA gene-disease ranking.

Why this file exists
--------------------
ULTRA's own script/run.py evaluates a held-out triple set and prints aggregate
metrics; it never writes per-instance scores, and its ranking is over all nodes
rather than over this project's fixed 4,399-gene candidate pool. Every LinkGDA
number is recomputed from a per-instance score file, so the baseline needs a
driver that produces one in the same layout.

The readout
-----------
The graph holds (gene, causes_phenotype, phenotype). The question asked of the
model is the same one evaluate_by_graph asks of TransD: how well does each
candidate gene explain each of the query disease's phenotypes. ULTRA scores a
whole tail set in one forward pass but only for a constant head and relation, so
the query is run in the reverse direction:

    head = phenotype, relation = inverse(causes_phenotype), tails = candidate genes

which is the same edge. ULTRA's loader appends inverse relations with index
r + num_direct_relations, and EntityNBFNet's negative_sample_to_tail leaves a
constant-head batch untouched, so this is a plain tail-prediction pass.

One pass per phenotype gives one score vector over the candidates. Diseases share
phenotypes heavily, so vectors are cached by phenotype and reused. Aggregation
over a disease's phenotype multiset then reproduces evaluate_by_graph exactly:

    gene_centric    = max  over the disease's phenotypes
    disease_centric = mean over the disease's phenotypes
    BMA = (gene_centric + disease_centric) / 2
    BMM = max(gene_centric, disease_centric)

The multiset is used, not the distinct set, because the mean in evaluate_by_graph
is taken over the phenotype rows as disease2pheno lists them.

Leakage
-------
ULTRA has no leakage guard at inference: whatever is in the message-passing graph
is available to the model. Only train.txt is used as that graph, and the export
already asserts no test disease appears in it. The assertion is repeated here
against the loaded vocabulary, since this is the last point where the graph the
model actually sees can be inspected.

A disease with no phenotype in the graph gets an all-zero score vector, matching
evaluate_by_graph's behaviour for the same case.
"""
import os
import re
import sys
import time
import json
import logging

import click as ck
import numpy as np
import torch
import yaml


logger = logging.getLogger("score_ultra")


def setup_logging(path):
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        handle = logging.FileHandler(path)
        handle.setFormatter(fmt)
        logger.addHandler(handle)


def load_model_config(ultra_root):
    """Model hyperparameters, read from ULTRA's own inference config.

    The file is a jinja template whose placeholders are not valid YAML, so the
    placeholders are blanked before parsing. Only the model section is used; the
    dataset, optimizer and training sections do not apply to a zero-shot pass.
    """
    path = os.path.join(ultra_root, "config", "transductive", "inference.yaml")
    with open(path) as handle:
        text = re.sub(r"\{\{.*?\}\}", "null", handle.read())
    return yaml.safe_load(text)["model"]


def build_dataset(ultra_root, data_root, name):
    sys.path.insert(0, ultra_root)
    from ultra.datasets import TransductiveDataset

    class LinkGDAFold(TransductiveDataset):
        """A fold graph exported by kge_transd.py --dump_triples.

        Nothing is downloaded: the raw files are written by prepare_ultra_data.py.
        Processing, inverse-edge construction and the relation graph all come from
        ULTRA's own TransductiveDataset, so the graph the model sees is built by
        ULTRA's code and not by a reimplementation of it.
        """

        delimiter = "\t"

        def __init__(self, root, name, **kwargs):
            self.name = name
            super().__init__(root, **kwargs)

        def download(self):
            raise RuntimeError(
                f"raw files for {self.name} are missing; run prepare_ultra_data.py first"
            )

    dataset = LinkGDAFold(root=data_root, name=name)
    vocab = dataset.load_file(dataset.raw_paths[0], inv_entity_vocab={}, inv_rel_vocab={})
    return dataset, vocab["inv_entity_vocab"], vocab["inv_rel_vocab"]


def read_test_pairs(path):
    pairs = []
    with open(path) as handle:
        for line in handle:
            disease, gene, phenos = line.rstrip("\n").split("\t")
            pairs.append((disease, gene, phenos.split("|") if phenos else []))
    return pairs


def score_phenotypes(model, graph, pheno_ids, gene_ids, relation, batch_size, device):
    """One score vector over the candidate genes per phenotype, batched over phenotypes."""
    scores = np.zeros((len(pheno_ids), len(gene_ids)), dtype=np.float32)
    tails = torch.as_tensor(gene_ids, dtype=torch.long, device=device)
    done = 0
    size = batch_size
    batches = 0
    while done < len(pheno_ids):
        take = min(size, len(pheno_ids) - done)
        heads = torch.as_tensor(pheno_ids[done:done + take], dtype=torch.long, device=device)
        batch = torch.stack([
            heads.unsqueeze(1).expand(take, len(gene_ids)),
            tails.unsqueeze(0).expand(take, len(gene_ids)),
            torch.full((take, len(gene_ids)), relation, dtype=torch.long, device=device),
        ], dim=-1)
        try:
            with torch.no_grad():
                out = model(graph, batch)
        except torch.cuda.OutOfMemoryError:
            del batch
            torch.cuda.empty_cache()
            if size == 1:
                raise
            size = max(1, size // 2)
            logger.warning(f"out of memory; reducing phenotype batch to {size}")
            continue
        scores[done:done + take] = out.float().cpu().numpy()
        done += take
        batches += 1
        if batches % 50 == 0:
            logger.info(f"scored {done}/{len(pheno_ids)} phenotypes")
    return scores


@ck.command()
@ck.option("--ultra_root", required=True, help="Path to the ULTRA checkout")
@ck.option("--data_root", required=True, help="ULTRA dataset root holding <name>/raw")
@ck.option("--name", required=True, help="Dataset name under data_root")
@ck.option("--dump", required=True, help="Triple dump; its companion files define candidates and test pairs")
@ck.option("--ckpt", required=True, help="ULTRA checkpoint to score with")
@ck.option("--identifier", required=True, help="Result file identifier; must encode every setting")
@ck.option("--out_dir", default="data/results", help="Where the score files are written")
@ck.option("--batch_size", type=int, default=8, help="Phenotypes scored per forward pass")
@ck.option("--log", default=None, help="Log file path")
@ck.option("--force_overwrite", is_flag=True, help="Allow writing over an existing result file")
def main(ultra_root, data_root, name, dump, ckpt, identifier, out_dir, batch_size, log, force_overwrite):
    setup_logging(log)
    started = time.time()

    prefix = os.path.join(out_dir, f"kge_results_{identifier}")
    bma_path, bmm_path = f"{prefix}_by_graph_bma.tsv", f"{prefix}_by_graph_bmm.tsv"
    if os.path.exists(bma_path) and not force_overwrite:
        raise SystemExit(f"refusing to overwrite {bma_path}; pass --force_overwrite to replace it")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"identifier {identifier}")
    logger.info(f"checkpoint {ckpt}")

    dataset, entity_to_id, relation_to_id = build_dataset(ultra_root, data_root, name)
    graph = dataset[0]
    num_direct_rel = len(relation_to_id)
    if graph.num_relations != 2 * num_direct_rel:
        raise SystemExit(f"relation count mismatch: vocabulary has {num_direct_rel} relations "
                         f"but the graph declares {graph.num_relations} including inverses; "
                         f"valid.txt or test.txt is missing a relation")
    logger.info(f"graph: {graph.num_nodes} nodes, {graph.num_edges} edges (inverses included), "
                f"{num_direct_rel} direct relations")

    if "causes_phenotype" not in relation_to_id:
        raise SystemExit("relation causes_phenotype absent from the exported graph")
    relation = relation_to_id["causes_phenotype"] + num_direct_rel
    logger.info(f"scoring relation: inverse(causes_phenotype) = {relation}")

    with open(f"{dump}.eval_genes.txt") as handle:
        eval_genes = [line.strip() for line in handle if line.strip()]
    missing = [g for g in eval_genes if g not in entity_to_id]
    if missing:
        raise SystemExit(f"{len(missing)} candidate genes absent from the graph, e.g. {missing[:5]}")
    gene_ids = [entity_to_id[g] for g in eval_genes]
    gene_to_index = {g: i for i, g in enumerate(eval_genes)}
    logger.info(f"candidates: {len(eval_genes)} genes, all present in the graph")

    test_pairs = read_test_pairs(f"{dump}.test_pairs.tsv")
    test_diseases = {d for d, _, _ in test_pairs}
    leaked = test_diseases & set(entity_to_id)
    if leaked:
        raise SystemExit(f"{len(leaked)} test disease(s) are nodes of the message-passing graph: {sorted(leaked)[:5]}")
    logger.info(f"leakage check passed: 0 of {len(test_diseases)} test diseases are graph nodes")

    queried = {p for _, _, phenos in test_pairs for p in phenos}
    distinct = sorted(p for p in queried if p in entity_to_id)
    pheno_row = {p: i for i, p in enumerate(distinct)}
    logger.info(f"test pairs: {len(test_pairs)}, distinct queried phenotypes: {len(distinct)} "
                f"({len(queried) - len(distinct)} of {len(queried)} absent from the graph and dropped)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, ultra_root)
    from ultra.models import Ultra

    cfg = load_model_config(ultra_root)
    model = Ultra(rel_model_cfg=cfg["relation_model"], entity_model_cfg=cfg["entity_model"])
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    graph = graph.to(device)
    logger.info(f"device {device}, parameters {sum(p.numel() for p in model.parameters())}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    scoring_started = time.time()
    matrix = score_phenotypes(model, graph, [entity_to_id[p] for p in distinct],
                              gene_ids, relation, batch_size, device)
    scoring_seconds = time.time() - scoring_started
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
    logger.info(f"scored {len(distinct)} phenotypes in {scoring_seconds:.1f}s "
                f"({scoring_seconds / max(len(distinct), 1):.3f}s each), peak GPU memory {peak:.2f} GiB")

    np.savez_compressed(f"{prefix}_phenotype_scores.npz",
                        phenotypes=np.array(distinct), genes=np.array(eval_genes), scores=matrix)

    bma_rows, bmm_rows, empty, degenerate = [], [], 0, 0
    for disease, gene, phenos in test_pairs:
        rows = [pheno_row[p] for p in phenos if p in pheno_row]
        if not rows:
            empty += 1
            zero = [0.0] * len(eval_genes)
            bma_rows.append((gene, disease, gene_to_index[gene], zero))
            bmm_rows.append((gene, disease, gene_to_index[gene], zero))
            continue
        block = matrix[rows]
        gene_centric = block.max(axis=0)
        disease_centric = block.mean(axis=0)
        bma = (gene_centric + disease_centric) / 2
        bmm = np.maximum(gene_centric, disease_centric)
        if float(bma.std()) == 0.0:
            degenerate += 1
        bma_rows.append((gene, disease, gene_to_index[gene], bma.tolist()))
        bmm_rows.append((gene, disease, gene_to_index[gene], bmm.tolist()))

    for path, rows in ((bma_path, bma_rows), (bmm_path, bmm_rows)):
        with open(path, "w") as handle:
            for gene, disease, index, scores in rows:
                handle.write(f"{gene}\t{disease}\t{index}\t" + "\t".join(str(s) for s in scores) + "\n")
        logger.info(f"wrote {path}")

    spreads = np.array([np.std(r[3]) for r in bma_rows])
    summary = {
        "identifier": identifier,
        "checkpoint": os.path.basename(ckpt),
        "nodes": int(graph.num_nodes),
        "edges": int(graph.num_edges),
        "direct_relations": int(num_direct_rel),
        "candidates": len(eval_genes),
        "test_pairs": len(test_pairs),
        "distinct_phenotypes": len(distinct),
        "pairs_without_phenotypes": empty,
        "degenerate_rows": degenerate,
        "bma_row_sd_min": float(spreads.min()),
        "bma_row_sd_median": float(np.median(spreads)),
        "scoring_seconds": scoring_seconds,
        "total_seconds": time.time() - started,
        "peak_gpu_gib": peak,
    }
    with open(f"{prefix}_run.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info(json.dumps(summary))


if __name__ == "__main__":
    main()
