"""Count production TransD graph stages without training a model."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from data import create_train_val_split


def rows(path):
    with open(path, newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        return list(csv.DictReader(handle, delimiter="\t" if sample.count("\t") > sample.count(",") else ","))


def edge_set(path):
    return {tuple(line.rstrip("\n").split("\t")) for line in open(path) if line.strip()}


def count_fold(data, fold, phenotypes, functions, sites, val_seed=0):
    ontology = edge_set(data / "upheno_edges.tsv")
    if functions:
        ontology |= edge_set(data / "go_edges.tsv")
    if sites:
        ontology |= edge_set(data / "uberon_edges.tsv")
    entities = {x for s, _, d in ontology for x in (s, d)}
    outer_train = pd.DataFrame(rows(data / "folds" / f"fold_{fold}" / "train.csv"))
    train, _ = create_train_val_split(outer_train, val_ratio=0.1, random_seed=val_seed)
    train = train.to_dict("records")
    test = rows(data / "folds" / f"fold_{fold}" / "test.csv")
    test_diseases = {r["Disease"] for r in test}
    disease = rows(data / "disease_phenotypes.csv")
    disease_edges = [(r["Disease"], r["Phenotype"]) for r in disease if r["Phenotype"] in entities and r["Disease"] not in test_diseases]
    entities |= {x for e in disease_edges for x in e}
    modalities = []
    if functions:
        modalities.append(("function", rows(data / "gene_functions.csv"), "Function"))
    if sites:
        modalities.append(("site", rows(data / "gene_site.csv"), "Tissue"))
    triples = list(ontology) + [(d, "has_symptom", p) for d, p in disease_edges]
    genes_other = set()
    for name, source, col in modalities:
        for r in source:
            if r[col] in entities:
                triples.append((r["Gene"], "has_function" if name == "function" else "expressed_in", r[col]))
                genes_other.add(r["Gene"])
    fallback = set()
    for r in rows(data / "gene_phenotypes.csv"):
        if r["Phenotype"] in entities and (phenotypes or r["Gene"] not in genes_other):
            triples.append((r["Gene"], "has_phenotype", r["Phenotype"]))
            if not phenotypes and r["Gene"] not in genes_other:
                fallback.add(r["Gene"])
                genes_other.add(r["Gene"])
    d2p = {}
    for d, p in disease_edges:
        d2p.setdefault(d, []).append(p)
    for r in train:
        triples.append((r["Gene"], "associated_with", r["Disease"]))
        for p in d2p.get(r["Disease"], []): triples.append((r["Gene"], "causes_phenotype", p))
    affected = [r for r in test if r["Gene"] in fallback]
    rel = Counter(r for _, r, _ in triples)
    unique = set(triples)
    return {"fold": fold, "entities": len({x for s, _, d in unique for x in (s,d)}), "relations": len(rel), "triple_rows": len(triples), "unique_triples": len(unique), "relation_triple_rows": dict(rel), "fallback_candidate_genes": len(fallback), "fallback_test_genes": len({r['Gene'] for r in affected}), "fallback_test_pairs": len(affected), "test_pairs": len(test)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data", type=Path, default=Path("data")); p.add_argument("--out", type=Path, required=True); a=p.parse_args()
    configs={"p":(True,False,False),"ps":(True,False,True),"pf":(True,True,False),"pfs":(True,True,True),"s":(False,False,True),"f":(False,True,False),"fs":(False,True,True)}
    result={k:[count_fold(a.data,i,*v) for i in range(10)] for k,v in configs.items()}
    a.out.write_text(json.dumps(result,indent=2)+"\n")

if __name__ == "__main__": main()
