# Controlled PhenoQuery comparison, fixed 28 October 2026

This protocol extends an exploratory benchmark whose test results already informed
method development. It uses the existing ten outer disease folds, inner training
and validation splits, frozen ontology inputs, and ordered 13,625-gene candidate
universe. Additional seeds measure training variability; the reused benchmark is
not an independent external validation cohort.

## Questions and fixed design

Use four arms: bridge_pool, plain_pool, plain_and, typed_and. Within each of two
initializations (ultra_4g.pth and ultraquery.pth), run seeds 0, 1, and 2 on every
fold: 240 training runs. All arms use the QueryNBFNet one-source projection and
identical optimization, validation and selection rules. The general checkpoint
remains the prespecified main PhenoQuery initialization; the query-pretrained
checkpoint is a controlled initialization comparison. Report every arm and seed.

The graph inputs are the retained, checksum-verified EQ campaign graphs. They
share gene annotations, materialized inner-training gene--phenotype links,
phenotype taxonomy and the common entity ontology backbone. All omit expression
edges. bridge_pool adds training disease nodes and their associated_with and
has_symptom edges. plain_pool and plain_and contain no disease nodes. typed_and
adds role-typed phenotype/entity links and distinct quality/qualifier links from
supported OWL expressions. Both graph orientations and the relation graph are
constructed before training; inference uses the corresponding prepared graph.

Pool is sigmoid-weighted max/mean over phenotype logits, initialized at weight
0.5. AND is mean log-sigmoid over unique phenotype branches. Full evaluation
profiles are retained. Each update uniformly samples a training disease and up
to eight distinct phenotypes, includes all its known positive candidate genes,
and uses 32 uniform negatives plus 32 hard negatives from 256 proposals.
Materialized causes_phenotype edges for all queried phenotypes are masked in
both directions, independently of candidate identity. In the bridge arm,
incident edges of training disease nodes sharing those phenotypes are also
masked. Relation-interaction graphs retain their training-only preparation.

Each run receives 500 updates with AdamW, learning rate 1e-5, zero weight decay,
gradient norm clipping at 1.0, and deterministic eval-mode graph forwards with
parameter gradients. Use exact two-pass chain-rule gradient accumulation.
Validate at step 0 and every 100 updates. Selection uses validation query-panel
leave-one-disease-out calibrated ranks. Retain step 0 as a fallback; an update is
eligible only if both Hits@1 and Hits@10 are at least their step-0 values. Among
eligible checkpoints maximize Hits@10, Hits@1, MRR, then negative mean rank.
Each fold makes its own checkpoint choice. No across-fold hyperparameter choice
uses outer-test results. Save the selected and initial checkpoints and their
validation score matrices. Optimizer and both NumPy/PyTorch RNG states are
checkpointed every ten updates; interruption/resumption preserves the trajectory.

Freeze all 240 selections before scoring any outer-test query in this campaign.
Evaluate both the selected and initial checkpoint for each arm. Record source,
input, checkpoint, graph, configuration and selection checksums. Validate all
candidate, query and positive-gene identities on import. No failed or incomplete
run contributes to a report.

## Endpoints and uncertainty

Primary endpoint: association-pooled Hits@10 using unique-test-disease
leave-one-out per-gene calibration. Retain native raw scores, mean rank, MRR,
and Hits@1 for every condition. Validation-reference calibration is an additional
explicit sensitivity if reported; its reference comprises the saved validation
score vectors, never test labels. Report initial-checkpoint performance alongside
selected-checkpoint performance to identify the effect of adaptation.

Four prespecified contrasts within each initialization: plain_pool minus
bridge_pool (training disease nodes), plain_and minus plain_pool (conjunction),
typed_and minus plain_and (typed EQ), and selected typed_and minus its initial
checkpoint (adaptation). These eight Hits@10 contrasts form one family. For each
association average the metric contribution across the three seeds, then use
10,000 paired phenotype-profile-cluster bootstrap replicates, seed 20261028,
with 99.375% two-sided percentile intervals (Bonferroni family alpha 0.05).
Cluster identical profiles across disease identifiers; preserve association
weights within each resampled cluster. Report each seed's estimate and the
between-seed spread separately. These intervals measure benchmark sampling
variation for the fitted models, not all training or selection uncertainty.
Other endpoint/initialization interactions are descriptive.

Bias analysis keeps score transformations and evaluation populations explicit.
Report seen/unseen status, same-gene training-phenotype overlap, and exact
training-profile matches. Analyze reference-profile deduplication, co-causal
filtering, equal-profile weighting, and training-profile exclusion separately.
A combined sensitivity may be reported with its exact denominator. Native
Phrank/PhenoRank rankings remain present alongside extra post-hoc adjustments.

## Technical gates and execution

Before production, run two-update fold-0 pilots for bridge_pool, plain_and and
typed_and under both initializations, using four validation diseases and no test
metrics. Verify one-source projection parity with the upstream ULTRA backbone,
finite nonzero gradients/parameter changes, candidate coverage and graph identity.
Verify a four-update typed_and run against the same run stopped after update 2
and resumed with the unchanged scientific configuration. Separately interrupt
after update 4 but before final validation, retain the exact interrupted state,
and compare that state with the resumed result. Validation must preserve weights,
optimizer and RNG bitwise, and complete the pending evaluation exactly once.
This isolates restoration from numerical variation between separately executed
GPU training trajectories. Selected checkpoint and
validation-score artifacts are restored from the same committed state as the
optimizer and RNG, including interruption during a later validation. Check RNG restoration,
selected step and parameter agreement to the documented floating-point tolerance.
Technical repairs before these gates pass are retained with their source hashes.

CPU preparation and reporting run on WS. GPU execution uses ORIX Slurm with
explicit H200 resources, modest concurrency respecting active account limits,
per-run locks, checkpointing and no automatic relaunch of terminal failures.
Scratch directories use a project path or TMPDIR, never /tmp.
