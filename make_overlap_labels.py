"""Write per-instance phenotype-overlap labels, so every method is stratified identically.

Overlap for a test instance (disease, gene) is the fraction of that disease's HPO
phenotypes already linked to the gene by a training causes_phenotype edge. The label
depends only on the fold and the training edges, never on the method being scored, so it
is computed once here and applied to every prediction file. That also lets the labels
travel to a host that holds prediction files but not the fold inputs.

The definition itself is imported from leakage_overlap.py rather than restated, so this
script cannot drift from the one the rest of the project uses. Three sources:

train-split  the 90% split kge_transd.py fits on, which is the project default and the
             edges the training graph actually carries.
full-train   every pair in the fold's train.csv, which is what the published stratified
             rows were computed from.
dump         the causes_phenotype triples read out of an exported graph. This needs no
             reconstruction at all, so it is the oracle the other two are checked against.

The denominator is the disease's full phenotype set in every case; only the numerator's
source changes.

Output: fold, disease, gene, overlap -- one row per test pair that has a phenotype set.
"""
from collections import defaultdict

import click as ck
import pandas as pd

from leakage_overlap import OVERLAP_SOURCES, disease_phenotypes, instance_overlap


def causes_phenotype_from_dump(path):
    cp = defaultdict(set)
    with open(path) as handle:
        for line in handle:
            src, rel, dst = line.rstrip("\n").split("\t")
            if rel == "causes_phenotype":
                cp[src].add(dst)
    return cp


@ck.command()
@ck.option("--source", type=ck.Choice(list(OVERLAP_SOURCES) + ["dump"]), default="train-split", show_default=True)
@ck.option("--dump-template", default=None, help="Graph dump path containing {fold}; required for --source dump")
@ck.option("--folds", default=10, show_default=True)
@ck.option("--out", required=True)
def main(source, dump_template, folds, out):
    if source == "dump" and not dump_template:
        raise SystemExit("--source dump requires --dump-template")

    if source != "dump":
        overlap = instance_overlap(source, folds)
        rows = [(f, d, g, v) for (f, d, g), v in overlap.items()]
    else:
        d2p = disease_phenotypes()
        rows = []
        for fold in range(folds):
            cp = causes_phenotype_from_dump(dump_template.format(fold=fold))
            test = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
            for gene, disease in zip(test["Gene"], test["Disease"]):
                phenos = d2p.get(disease, set())
                if not phenos:
                    continue
                rows.append((fold, disease, gene, len(phenos & cp.get(gene, set())) / len(phenos)))

    with open(out, "w") as handle:
        handle.write("fold\tdisease\tgene\toverlap\n")
        for fold, disease, gene, value in rows:
            handle.write(f"{fold}\t{disease}\t{gene}\t{value:.10g}\n")

    zero = sum(1 for r in rows if r[3] == 0)
    print(f"{out}: {len(rows)} labelled instances, {zero} at zero overlap "
          f"({100 * zero / len(rows):.1f}%), {len(rows) - zero} above zero "
          f"({100 * (len(rows) - zero) / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
