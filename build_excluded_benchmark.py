"""Build the excluded-gene benchmark.

The main benchmark keeps only gene--disease pairs whose gene carries at least one
MGI-propagated phenotype annotation (build_association_files.py:326-330). That filter
exists so every baseline can score every candidate: the semantic-similarity measures,
Exomiser-Phive and INDIGENA all match phenotype sets, and a gene with no phenotype
profile gets no score from any of them. The side effect is that the population the
method is motivated by -- genes whose mouse orthologs lack phenotype annotations --
never appears in the evaluation, so the phenotype-free result is obtained by
withholding annotations from genes that have them rather than by measuring genes that
do not.

This script builds the complementary benchmark from the pairs that filter discards.
A pair enters when its gene has no MGI phenotype but does have GO function
annotations, and its disease has HPO phenotypes.

Two leakage filters then apply, because the model represents a query disease only by
its HPO phenotype set:

  1. ID overlap. Diseases that also appear in the training pairs are dropped.
  2. Profile overlap. A disease whose phenotype set closely matches a training
     disease's is the same query under a different identifier, so diseases whose
     maximum Jaccard similarity to any training disease reaches --jaccard-threshold
     are dropped. The default of 0.5 is a choice, not a standard; the script reports
     counts at several thresholds so the sensitivity is visible.

Outputs, under --out-dir:

  data/folds/fold_0/train.csv   all kept pairs (training)
  data/folds/fold_0/test.csv    the excluded pairs surviving both leakage filters
  data/gene_diseases.csv        candidate pool: main pool plus the new test genes
  funnel.txt                    the construction counts, for the paper and README
  disease_leakage.csv           per-test-disease max Jaccard and nearest train disease

Shared inputs (ontologies, annotation CSVs) are symlinked from --data-dir rather than
copied, so the benchmark adds megabytes rather than gigabytes.

Run from the repository root:
    python build_excluded_benchmark.py --data-dir data --out-dir ../link-gda-excluded
"""
from collections import defaultdict
import os

import click as ck
import pandas as pd

# Files kge_transd.py reads relative to its working directory. Symlinked, not copied.
# The projected edge lists are what the trainer actually consumes; the ontologies are
# only a fallback it uses to regenerate them, so a host without the OWL files still
# trains as long as the edge lists are present.
SHARED_INPUTS = [
    "disease_phenotypes.csv",
    "gene_phenotypes.csv",
    "gene_functions.csv",
    "gene_site.csv",
    "genes_to_disease.txt",
    "upheno_edges.tsv",
    "upheno_edges_gda.tsv",
    "upheno_edges_gda_hp_only.tsv",
    "upheno_owl2vecstar_edges.tsv",
    "go_edges.tsv",
    "uberon_edges.tsv",
    "upheno.owl",
    "go.owl",
    "go-plus.owl",
    "mp.owl",
    "uberon.owl",
]


def load_pairs(data_dir):
    """Gene--disease pairs from genes_to_disease.txt, keyed as (entrez, disease_id)."""
    gd = pd.read_csv(os.path.join(data_dir, "genes_to_disease.txt"), sep="\t")
    gd.columns = ["ncbi_gene_id", "gene_symbol", "association_type", "disease_id", "source"]
    return {(r.ncbi_gene_id.split(":")[1], r.disease_id) for r in gd.itertuples()}


def load_gene_set(data_dir, filename):
    """Bare gene identifiers from the first column of an annotation CSV."""
    df = pd.read_csv(os.path.join(data_dir, filename))
    return {str(g).split("/")[-1] for g in df.iloc[:, 0]}


def load_disease_phenotypes(data_dir):
    """Map disease id (OMIM:123456 form) to its set of HPO terms."""
    dp = pd.read_csv(os.path.join(data_dir, "disease_phenotypes.csv"))
    d2p = defaultdict(set)
    for disease, phenotype in zip(dp.iloc[:, 0], dp.iloc[:, 1]):
        d2p[str(disease).split("/")[-1].replace("_", ":")].add(str(phenotype))
    return d2p


def max_jaccard_to_training(test_diseases, train_diseases, d2p):
    """For each test disease, its closest training disease by Jaccard on HPO sets.

    Candidates come from an inverted index, so only training diseases sharing at
    least one phenotype are compared.
    """
    phenotype_to_disease = defaultdict(set)
    for disease in train_diseases:
        for phenotype in d2p.get(disease, ()):
            phenotype_to_disease[phenotype].add(disease)

    rows = []
    for disease in sorted(test_diseases):
        profile = d2p.get(disease, set())
        if not profile:
            continue
        candidates = set()
        for phenotype in profile:
            candidates |= phenotype_to_disease.get(phenotype, set())
        best, nearest = 0.0, None
        for candidate in candidates:
            other = d2p.get(candidate, set())
            similarity = len(profile & other) / len(profile | other)
            if similarity > best:
                best, nearest = similarity, candidate
        rows.append((disease, len(profile), best, nearest))
    return pd.DataFrame(rows, columns=["disease", "n_phenotypes", "max_jaccard", "nearest_train"])


@ck.command()
@ck.option("--data-dir", default="data", help="Main benchmark data directory")
@ck.option("--out-dir", required=True, help="Directory to build the new benchmark in")
@ck.option("--jaccard-threshold", default=0.5, help="Drop test diseases at or above this similarity")
def main(data_dir, out_dir, jaccard_threshold):
    data_dir = os.path.abspath(data_dir)
    out_dir = os.path.abspath(out_dir)
    out_data = os.path.join(out_dir, "data")
    os.makedirs(os.path.join(out_data, "folds", "fold_0"), exist_ok=True)
    os.makedirs(os.path.join(out_data, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_data, "results"), exist_ok=True)

    pairs = load_pairs(data_dir)
    with_phenotypes = load_gene_set(data_dir, "gene_phenotypes.csv")
    with_functions = load_gene_set(data_dir, "gene_functions.csv")
    d2p = load_disease_phenotypes(data_dir)

    # The main benchmark is gene_diseases.csv, the output of both filters in
    # build_association_files.py (gene has a phenotype, and disease has phenotypes).
    # Take the training pairs and the leakage reference from it directly rather than
    # recomputing, so the two benchmarks cannot drift apart.
    main_pool = pd.read_csv(os.path.join(data_dir, "gene_diseases.csv"))
    kept = {(str(g).split("/")[-1], str(d).split("/")[-1].replace("_", ":"))
            for g, d in zip(main_pool["Gene"], main_pool["Disease"])}

    discarded = {p for p in pairs if p[0] not in with_phenotypes}
    scoreable = {p for p in discarded if p[0] in with_functions and p[1] in d2p}

    train_diseases = {p[1] for p in kept}
    id_clean = {p for p in scoreable if p[1] not in train_diseases}

    leakage = max_jaccard_to_training({p[1] for p in id_clean}, train_diseases, d2p)
    leaky = set(leakage[leakage.max_jaccard >= jaccard_threshold].disease)
    test_pairs = sorted(p for p in id_clean if p[1] not in leaky)

    # Candidate pool: the main pool plus the genes introduced by this test set, so
    # every true gene is rankable. Mean ranks are therefore over a slightly larger
    # pool than the main benchmark and are not directly comparable to it.
    pool_genes = {str(g).split("/")[-1] for g in main_pool["Gene"]}
    new_genes = {p[0] for p in test_pairs}

    def gene_uri(g):
        return f"http://mowl.borg/{g}"

    def disease_uri(d):
        return f"http://mowl.borg/{d.replace(':', '_')}"

    extended = pd.concat([
        main_pool,
        pd.DataFrame([{"Gene": gene_uri(g), "Disease": disease_uri(d)} for g, d in test_pairs]),
    ], ignore_index=True)
    extended.to_csv(os.path.join(out_data, "gene_diseases.csv"), index=False)

    fold_dir = os.path.join(out_data, "folds", "fold_0")
    pd.DataFrame([{"Gene": gene_uri(g), "Disease": disease_uri(d)} for g, d in sorted(kept)]).to_csv(
        os.path.join(fold_dir, "train.csv"), sep="\t", index=False)
    pd.DataFrame([{"Gene": gene_uri(g), "Disease": disease_uri(d)} for g, d in test_pairs]).to_csv(
        os.path.join(fold_dir, "test.csv"), sep="\t", index=False)
    leakage.to_csv(os.path.join(out_dir, "disease_leakage.csv"), index=False)

    for name in SHARED_INPUTS:
        source = os.path.join(data_dir, name)
        link = os.path.join(out_data, name)
        if os.path.exists(source) and not os.path.lexists(link):
            os.symlink(source, link)

    lines = [
        "Excluded-gene benchmark construction",
        "=" * 60,
        f"jaccard threshold                                  : {jaccard_threshold}",
        "",
        f"all pairs in genes_to_disease.txt                  : {len(pairs):>7,}",
        f"  main benchmark, after both filters               : {len(kept):>7,}",
        f"  gene has NO MGI phenotype (discarded by filter)  : {len(discarded):>7,}",
        f"    and gene has GO functions, disease has HPO     : {len(scoreable):>7,}",
        f"    and disease id not in training                 : {len(id_clean):>7,}",
        f"    and phenotype profile not near-duplicate       : {len(test_pairs):>7,}  <- test set",
        "",
        f"test genes    : {len(new_genes):>6,}",
        f"test diseases : {len({p[1] for p in test_pairs}):>6,}",
        f"candidate pool: {len(pool_genes):,} main + {len(new_genes - pool_genes):,} new "
        f"= {len(pool_genes | new_genes):,}",
        "",
        "Leakage sensitivity (of the id-disjoint candidates):",
    ]
    for threshold in (1.0, 0.9, 0.8, 0.5, 0.3):
        n = int((leakage.max_jaccard >= threshold).sum())
        lines.append(f"  max jaccard >= {threshold:<4} : {n:>4} diseases dropped "
                     f"({n / max(len(leakage), 1):.1%})")
    lines.append(f"  median max jaccard  : {leakage.max_jaccard.median():.3f}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "funnel.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
