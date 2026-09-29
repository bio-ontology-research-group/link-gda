"""Audit the exact graph assembled by kge_transd.py without training a model."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from data import create_train_val_split


VARIANTS = {"p": (True, False, False), "ps": (True, False, True), "pf": (True, True, False), "pfs": (True, True, True), "s": (False, False, True), "f": (False, True, False), "fs": (False, True, True)}


def table(path, sep=","):
    return pd.read_csv(path, sep=sep, dtype=str)


def edge_rows(path):
    result = []
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{number}: expected three tab-separated fields")
            result.append(tuple(fields))
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def snapshot(triples):
    unique = set(triples)
    row_relations = Counter(relation for _, relation, _ in triples)
    unique_relations = Counter(relation for _, relation, _ in unique)
    entities = {entity for head, _, tail in unique for entity in (head, tail)}
    return {"entities": len(entities), "external_relation_types": len(unique_relations), "loaded_triple_rows": len(triples), "unique_triples": len(unique), "duplicate_rows": len(triples) - len(unique), "loaded_rows_by_relation": dict(sorted(row_relations.items())), "unique_triples_by_relation": dict(sorted(unique_relations.items()))}


def assemble(data, fold, variant, projector="owl2vecstar", val_seed=0):
    use_phenotypes, use_functions, use_sites = VARIANTS[variant]
    projection_name = "upheno_edges_gda.tsv" if projector == "owl2vecstar_gda" else "upheno_edges.tsv"
    input_paths = [data / projection_name]
    if use_functions:
        input_paths.append(data / "go_edges.tsv")
    if use_sites:
        input_paths.append(data / "uberon_edges.tsv")
    triples = []
    entities = set()
    for path in input_paths:
        for head, relation, tail in edge_rows(path):
            triples.append((head, relation, tail))
            entities.update((head, tail))
    stages = {"ontology_projection": snapshot(triples)}
    outer_train = table(data / "folds" / f"fold_{fold}" / "train.csv", "\t")
    train, validation = create_train_val_split(outer_train, val_ratio=0.1, random_seed=val_seed)
    test = table(data / "folds" / f"fold_{fold}" / "test.csv", "\t")
    test_diseases = set(test.Disease)
    disease_phenotypes = table(data / "disease_phenotypes.csv")
    disease_annotation_rows = 0
    missing_disease_annotation_rows = 0
    for row in disease_phenotypes.itertuples(index=False):
        if row.Phenotype not in entities:
            missing_disease_annotation_rows += 1
            continue
        if row.Disease not in test_diseases:
            triples.append((row.Disease, "has_symptom", row.Phenotype))
            entities.add(row.Disease)
            disease_annotation_rows += 1
    modality_rows = {"has_symptom": disease_annotation_rows}
    if use_functions:
        used = 0
        for row in table(data / "gene_functions.csv").itertuples(index=False):
            if row.Function in entities:
                triples.append((row.Gene, "has_function", row.Function))
                entities.update((row.Gene, row.Function))
                used += 1
        modality_rows["has_function"] = used
    if use_sites:
        used = 0
        for row in table(data / "gene_site.csv").itertuples(index=False):
            if row.Tissue in entities:
                triples.append((row.Gene, "expressed_in", row.Tissue))
                entities.update((row.Gene, row.Tissue))
                used += 1
        modality_rows["expressed_in"] = used
    fallback_genes = set()
    gene_pheno_rows = 0
    missing_gene_pheno_rows = 0
    for row in table(data / "gene_phenotypes.csv").itertuples(index=False):
        if row.Phenotype not in entities:
            missing_gene_pheno_rows += 1
        elif use_phenotypes or row.Gene not in entities:
            triples.append((row.Gene, "has_phenotype", row.Phenotype))
            if not use_phenotypes:
                fallback_genes.add(row.Gene)
            entities.add(row.Gene)
            gene_pheno_rows += 1
    modality_rows["has_phenotype"] = gene_pheno_rows
    stages["annotations"] = snapshot(triples)
    non_test_diseases = set(train.Disease) | set(validation.Disease)
    overlap = test_diseases & non_test_diseases
    if overlap:
        raise AssertionError(f"test diseases overlap train/validation: {sorted(overlap)[:5]}")
    if test_diseases & entities:
        raise AssertionError("test diseases occur among graph entities before training edges")
    for row in train.itertuples(index=False):
        triples.append((row.Gene, "associated_with", row.Disease))
        if row.Disease not in entities:
            raise AssertionError(f"training disease absent after annotations: {row.Disease}")
    stages["training_associations"] = snapshot(triples)
    disease2pheno = {}
    for row in disease_phenotypes.itertuples(index=False):
        if row.Phenotype in entities:
            disease2pheno.setdefault(row.Disease, []).append(row.Phenotype)
    causes_rows = 0
    for row in train.itertuples(index=False):
        for phenotype in disease2pheno.get(row.Disease, []):
            triples.append((row.Gene, "causes_phenotype", phenotype))
            causes_rows += 1
    stages["causes_phenotype"] = snapshot(triples)
    final = stages["causes_phenotype"]
    candidate_genes = set(table(data / "gene_diseases.csv").Gene)
    fallback_candidates = fallback_genes & candidate_genes
    affected = test[test.Gene.isin(fallback_candidates)]
    true_genes = set(test.Gene)
    final_entities = {entity for head, _, tail in triples for entity in (head, tail)}
    leaked_test_diseases = test_diseases & final_entities
    if leaked_test_diseases:
        raise AssertionError(f"test diseases occur in final triples: {sorted(leaked_test_diseases)[:5]}")
    missing_candidates = candidate_genes - final_entities
    final.update({"internal_relation_types_with_inverses": 2 * final["external_relation_types"], "loaded_training_instances_with_inverses": 2 * final["loaded_triple_rows"], "unique_training_instances_with_inverses": 2 * final["unique_triples"]})
    inputs = {str(path.relative_to(data)): digest(path) for path in input_paths}
    for name in ("disease_phenotypes.csv", "gene_phenotypes.csv", "gene_diseases.csv", f"folds/fold_{fold}/train.csv", f"folds/fold_{fold}/test.csv"):
        inputs[name] = digest(data / name)
    if use_functions:
        inputs["gene_functions.csv"] = digest(data / "gene_functions.csv")
    if use_sites:
        inputs["gene_site.csv"] = digest(data / "gene_site.csv")
    return {"fold": fold, "variant": variant, "projector": projector, "split": {"outer_train_pairs": len(outer_train), "training_pairs": len(train), "validation_pairs": len(validation), "test_pairs": len(test), "test_disease_overlap": 0}, "stages": stages, "annotation_rows_added": modality_rows, "missing_annotation_rows": {"disease_phenotype": missing_disease_annotation_rows, "gene_phenotype": missing_gene_pheno_rows}, "causes_phenotype_rows_added": causes_rows, "evaluation_pool": {"candidate_genes": len(candidate_genes), "candidate_genes_absent_from_graph": len(missing_candidates), "test_genes": len(true_genes), "test_pairs": len(test)}, "phenotype_fallback": {"eligible_genes_in_all_annotations": len(fallback_genes), "candidate_genes_affected": len(fallback_candidates), "candidate_genes_denominator": len(candidate_genes), "true_test_genes_affected": len(set(affected.Gene)), "true_test_genes_denominator": len(true_genes), "test_pairs_affected": len(affected), "test_pairs_denominator": len(test)}, "input_sha256": dict(sorted(inputs.items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+")
    parser.add_argument("--projector", choices=("owl2vecstar", "owl2vecstar_gda"), default="owl2vecstar")
    parser.add_argument("--val-seed", type=int, default=0)
    args = parser.parse_args()
    folds = args.folds if args.folds is not None else sorted(int(path.name.removeprefix("fold_")) for path in (args.data / "folds").glob("fold_*"))
    variants = args.variants or list(VARIANTS)
    result = {"schema_version": 1, "data_directory": str(args.data.resolve()), "val_seed": args.val_seed, "counts": [assemble(args.data, fold, variant, args.projector, args.val_seed) for variant in variants for fold in folds]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
