# Matched KGE graph sensitivity, fixed 29 October 2026

This rerun replaces the unmatched original/no-bridge configurations when
estimating the effect of removing training disease bridges. The benchmark
already informed method development. The experiment is a controlled sensitivity
on the reused ten-fold benchmark, not independent external validation.

For each of ten folds, train five model families on two graphs: 100 training
runs. `original` uses the frozen inner-training graph; `nobridge` removes
`associated_with` and `has_symptom` triples and rebuilds the entity/relation
vocabulary. All training disease nodes must disappear from the second graph;
validation and test disease nodes must be absent from both. Candidate genes,
annotation sources, ontology snapshots, materialized training gene–phenotype
links and disease splits stay fixed. Changing graph size changes the embedding
inventory and initialization mapping; this is a graph-construction sensitivity,
not a same-parameter inference-time edge mask.

The same implementation and scientific settings apply to both graphs:

| Family | Dimension | Training | Primary validation choice |
|---|---:|---|---|
| U1 | 200 | TransD, one BasicNegativeSampler corruption | Calibrated mean rank |
| T1 | 200 | TransD, one candidate-gene corruption for the phenotype relation; reject known positives | Raw Hits@10, then Hits@1, MRR, negative mean rank |
| INDIGENA | 400 | TransD with the U1 training objective; phenotype-embedding BMA readout | Raw mean rank |
| ConvKB-D | 200 | Warm start from the same graph's U1 calibrated-MR choice; 100 filters, no hidden dropout | Calibrated mean rank |
| P-pool | 200 | Warm start from the same graph's U1 raw-head choice; update encoder and learned pooling weight | Raw head metrics |

Seed is 0. Graph models receive 1,000 complete training epochs with Adam,
batch size 65,536, subbatch size 8,192 and no early stopping. Learning rate is
0.001 for TransD and 0.0001 for ConvKB. Validate every 20 epochs; the final
epoch is included. Epoch budgets and batch rules are matched, but the smaller
graph has fewer training triples and may require fewer updates per epoch.
Report actual graph sizes and epoch budgets; do not call this an equal-update
or equal-FLOP comparison.

P-pool receives 20 epochs with Adam at 0.0001, accumulation over 16 training
diseases, temperature 1, 32 uniform and 32 hard negatives from 256 proposals.
Its sampler rejects known training positives for the current disease.
Validation disease identities are checked for split disjointness; validation
labels do not define the negative pool. Validate
every epoch. The same procedure applies on each graph.

Retain four choices for every trajectory: raw/calibrated mean rank and
raw/calibrated head metrics. A strict improvement replaces a checkpoint;
ties retain the earlier epoch. These choices are fixed before test scoring.
Report the specified primary choice without selecting it from test outcomes;
the other choices are explicitly secondary sensitivities.

Save complete optimizer/RNG/validation states and atomic selected-checkpoint
snapshots. Record source hashes, raw data identities, mappings, actual graph
size, dimension, negative sampler, seed and warm-start checkpoint. Reject
incompatible resumption. Technical interruption is separate from scientific
stopping and must not export partial test results.

Freeze all 100 `TRAINED.json` records and selected checkpoint hashes globally
before any test-score export. The freeze verifies identical settings, inputs,
source and environments within each graph pair. Export complete candidate
scores for test and training disease profiles, with explicit identity axes.
Score transformations and cohort sensitivities use the shared score evaluator.
For graph-pair comparisons, report paired profile-cluster bootstrap intervals
for each family's primary Hits@10 and mean rank, with the multiple-comparison
family stated explicitly. Do not fold these new comparisons into the older
unmatched results under the same label.

Before production, exercise all five families and both graphs on a small real
CPU fixture, including warm-start dependencies, interrupted resumption,
incomplete-freeze rejection, pre-freeze score rejection and completed-score
checksum rejection. Then run full-graph GPU training/validation pilots before
freezing an immutable production source snapshot.
