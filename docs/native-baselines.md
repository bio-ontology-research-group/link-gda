# Native baseline algorithms and declared adaptations

The primary Phrank and PhenoRank results retain each method's native scoring.
Additional calibration is reported separately. Both methods refit human
annotations using only the fold's inner-training disease–gene associations.
Every candidate remains in the evaluation, including genes with no evidence.

## Phrank: native HPO direct-gene score

Install `phrank==1.6`. `--author-root` names a directory containing its `phrank/`
package, either an unpacked source distribution or the installation's
`site-packages` directory. The adapter records hashes of the actual author
implementation. It caches closures and uses stable summation of marginal
information content; sampled real-query scores and fixture rankings are checked
against the author implementation.

```bash
python code/baselines/native_phrank/run.py extract \
  --ontology INPUTS/upheno.owl --output WORKSPACE/native-hpo
python code/baselines/native_phrank/run.py run \
  --data-root WORKSPACE/data --ontology-root WORKSPACE/native-hpo \
  --author-root PHRANK_SOURCE --output-root WORKSPACE/native-phrank --fold 0
```

Repeat `run` for folds 0–9. Existing fold outputs are protected from overwriting.
The ontology extraction keeps explicit HPO-to-HPO subclass axioms from the
retained UPheno OWL snapshot; it excludes cross-species and restriction edges.
Unknown query terms remain singleton nodes. Information content and gene
annotations use inner-training associations. Unannotated genes have zero scores
and average-rank ties. The score formula is native Phrank; the whole-candidate,
fold-specific task differs from the published exome benchmark.

The [frozen protocol](../code/baselines/native_phrank/PROTOCOL.md) specifies the
ontology and annotation conventions. Outputs include full scores, association
identities, metrics, annotation provenance, hashes and author-parity counts.

The published Phrank gene score takes the maximum over associated diseases.
Run this additional mode with the same frozen inputs and a separate output root:

```bash
python code/baselines/native_phrank/run.py run --mode disease-max \
  --data-root WORKSPACE/data --ontology-root WORKSPACE/native-hpo \
  --author-root PHRANK_SOURCE --output-root WORKSPACE/native-phrank-disease-max --fold 0
```

Its [protocol](../code/baselines/native_phrank/DISEASE-MAX-PROTOCOL.md) fixes
training-only disease references, native raw scoring and author-API parity.
Report both modes by name; their different definitions can change gene rankings.

## PhenoRank: published simulation and historical-resource adapter

```bash
python code/baselines/native_phenorank/run.py \
  --root WORKSPACE --data-root WORKSPACE/data \
  --resources INPUTS/phenorank-resources \
  --output-root WORKSPACE/native-phenorank --fold 0
```

The resources directory contains `hgnc_complete_set.txt` and the archived
`original/` resources listed in each fold's resource audit. Preserve their
release checksums: live replacements change the experiment. The source port
retains 20 random-walk iterations, restart probability 0.1, rank comparison
against 1,000 simulated queries, and the empirical-p floor 1/1000. Co-occurrence
sampling fills decreasing probability tiers, including sampling with replacement
inside partially used tiers. The upstream source is commit
`9689f593b8a59650d7fa8525af95045744511f6e`; its BSD license is retained in
[`third_party/PhenoRank-LICENSE.txt`](../third_party/PhenoRank-LICENSE.txt).

The historical mouse and PPI evidence is retained. Human annotations, information
content and simulation co-occurrence are rebuilt per fold. Ontology-based mapping
to the archived phenotype vocabulary, candidate-universe extension and the
no-evidence convention are explicit adaptations, described in the
[frozen protocol](../code/baselines/native_phenorank/PROTOCOL.md). Every mapping
and unsupported profile is recorded. Per-query chunks support source-identical
resumption. Primary scores are negative empirical p-values; raw propagation
scores are saved separately.

## UPheno and cross-species adaptations

`run_matched_phrank_phenorank.py` implements separate UPheno-based adaptations,
including human-only, human+mouse, and mouse-only conditions. Its human-only
Phrank uses UPheno ancestry and is distinct from the native HPO adapter above.
Parentless terms use the author's information-content denominator: the number
of genes with nonempty annotations. This matters when human and mouse branches
cover different gene sets. `--phrank-root-policy legacy-zero` explicitly
reproduces the historical adaptation that suppressed all root contributions.
The UPheno matrices retain their float32 arithmetic; the HPO adapter above uses
float64 with stable summation. See the [root-information protocol](../reproducibility/protocols/upheno-phrank-20261029.md).
The published-tier sampler and uncapped profile lengths are the default;
`--sampler legacy-weighted` reproduces the superseded weighted/capped experiment.
Null queries are shared by query length in these UPheno adaptations. The native
historical-resource adapter instead simulates independently for every query.

```bash
python code/baselines/run_matched_phrank_phenorank.py \
  --base-dir WORKSPACE --phenorank-resources INPUTS/phenorank-resources \
  --out-root WORKSPACE/upheno-baselines --folds 0 1 2 3 4 5 6 7 8 9 --workers 2
```

`WORKSPACE/data/foldN` must contain the frozen prepared splits, candidates,
profiles and graph. This layout uses the workspace as `--base-dir`, not its
`data` subdirectory. Input-archive publication and clean-download verification
are tracked in [reproduction status](../reproducibility/STATUS.txt).

Use `--phrank-only` to generate the ten Phrank variants in a fresh directory
without rerunning independent PhenoRank simulations. Completion metadata records
the root convention and exact model coverage. Never mix root conventions under
one result label.
