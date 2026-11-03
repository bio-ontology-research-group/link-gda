# Compositional gene–disease association, Track A

Track A asks whether query-embedding models trained only on one-hop facts can
rank candidate genes for a held-out disease by composing two learned
projections through a latent disease. The training graph contains no
materialised gene-to-phenotype shortcut, and evaluation never looks up the
held-out disease as an entity; it enters only through its phenotype profile.

The package `code/compositional_gda/` implements four method families and five
variants:

| method key | model | source |
|---|---|---|
| `transe` | GQE/TransE path: anchor vector plus each relation vector, L1 score | KGReasoning `geo=vec` |
| `q2b` | Query2Box: point entities, box queries, unclamped relation offsets (`offset_activation=none`) | KGReasoning `geo=box` |
| `betae` | BetaE: Beta embeddings, MLP projection, KL score | KGReasoning `geo=beta` |
| `geometre` + `membership_mode=original` | GeometrE released code, transitivity off, membership loss softplus(−d) | GeometrE `src/` |
| `geometre` + `membership_mode=sign_corrected` | identical except the membership loss is softplus(d) | experimental sign correction |

The pinned source commits are recorded in `models.SOURCES` and in every run's
`identity.json`. The sign-corrected variant changes only the sign inside the
membership log-sigmoid. It is not a reconstruction of a formula from the
GeometrE paper.

## Requirements

Python 3.12 with the pinned packages in `requirements-compositional.txt`
(torch 2.6.0, numpy 2.2.6, pytest 8.3.5). For GPU runs, install the torch 2.6.0
wheel built for your CUDA version, for example from
`https://download.pytorch.org/whl/cu124`. The installed versions are part of a
run's identity, so a resumed run must use the same versions.

## Input: a prepared fold

`train.py` reads a directory written by `code/compositional_gda/prepare.py`:
`graph.npz` (`triples`, int64, shape N×3, inverse triples included),
`vocabulary.json`, `profiles.json`, `candidates.txt`, `training.tsv`,
`validation.tsv`, `test.tsv` and `identity.json` with each file's SHA-256 under
`outputs`. The trainer refuses to start when a digest differs, when
`causes_phenotype` is present, when a validation or test disease is an entity,
or when a candidate or validation phenotype is missing from the vocabulary.
`test.tsv` is read only for that disjointness check.

## Configuration

A run is described by one JSON file. Only `data_dir` and `method` are required;
GeometrE also requires `membership_mode`. A relative `data_dir` is resolved
against the directory of the configuration file. Unknown keys, and keys that do
not apply to the chosen method, are refused.

| key | default | meaning |
|---|---|---|
| `seed` | 0 | model initialisation and sampler seed |
| `steps` | 500 | fixed number of updates |
| `eval_every` | 100 | validation cadence; step 0 and the final step are always validated |
| `dim` | 100 | embedding dimension |
| `batch_size` | 128 | one-hop queries per update |
| `negative_sample_size` | 32 | negatives per query |
| `learning_rate` | 1e-4 | Adam, constant |
| `validation_diseases` | 32 | first N validation diseases in sorted order; `null` uses all |
| `aggregation_weight` | 0.5 | weight of the maximum in max/mean profile aggregation |
| `query_chunk`, `score_chunk_elements` | 1024, 2^25 | scoring memory bound |
| `checkpoint_every_steps`, `checkpoint_every_seconds` | 100, 900 | periodic commits between validations |
| `deterministic` | true | `torch.use_deterministic_algorithms`; TF32 is always off |
| `name` | null | free label stored in the identity |
| `auxiliary` | null | null for Track A; `two_hop` or `one_hop_readout` for the [two-hop training diagnostic](#two-hop-training-diagnostic) |
| `auxiliary_slots` | 0 | queries per update taken from the auxiliary groups; must be 0 when `auxiliary` is null |

Method defaults: `transe` gamma 24; `q2b` gamma 24, `cen` 0.02,
`offset_activation` `none`; `betae` gamma 60, `beta_hidden` 1600,
`beta_layers` 2; `geometre` gamma 20, `alpha` 0.2, `answer_embedding` false.

Example, `configs/geometre-original-fold0.json`:

```json
{"data_dir": "../prepared/fold0", "method": "geometre", "membership_mode": "original"}
```

## Training and evaluation

Every (head, relation) pair of the graph, inverse relations included, is a
one-hop training query whose answers are all its known tails. Each update draws
a batch in a per-epoch shuffled order, one uniform positive per query, and
negatives drawn uniformly with the query's known answers removed. The loss is
the KGReasoning negative-sampling loss with subsampling weight
1/sqrt(4 + number of answers). GeometrE adds its membership term.

Validation scores two readouts with the same weights:

- **forward**: gene → `associated_with` → `has_symptom`, scored against each
  profile phenotype;
- **inverse**: phenotype → `inverse:has_symptom` → `inverse:associated_with`,
  scored against each candidate gene.

Each readout gives a phenotype-by-gene matrix over all frozen candidates. A
disease's score for a gene is 0.5 × maximum + 0.5 × mean over its profile, with
duplicate profile entries kept. This differs from PhenoQuery
(`code/phenoquery/common.py`), which pools over the distinct phenotypes of a
profile: on a profile with a repeated phenotype the mean term differs, so
aggregated scores, ranks and metrics are not comparable between the two studies
for such diseases. Calibration, ranking, metrics and selection are the same.
Calibrated scores use leave-one-disease-out
per-gene standardisation over the validation panel, and ranks use average-rank
ties. Each readout selects its own checkpoint from the same trajectory: maximise
calibrated Hits@10, then Hits@1, MRR and negative MR; step 0 is the fallback; a
later step is eligible only if its calibrated Hits@1 and Hits@10 are at least
the step-0 values; an exact tie keeps the earlier step.

## Two-hop training diagnostic

This diagnostic is separate from Track A, whose models see only one-hop
queries, and from Track B's ontology queries. It asks whether explicit training
on the composed readout paths changes how the models fit and rank. It adds no
fact and no relation to the graph. Every two-hop answer is a join of one-hop
training edges, but training on those answers is still explicit derived
supervision, so the training task differs from Track A's.

With `auxiliary` set, each update keeps `batch_size` queries:
`batch_size - auxiliary_slots` come from the default one-hop stream and the rest
are split evenly over the auxiliary groups.

| `auxiliary` | groups (in batch order after the one-hop group) |
|---|---|
| `two_hop` | `forward_2p`: gene → `associated_with` → `has_symptom`; `inverse_2p`: phenotype → `inverse:has_symptom` → `inverse:associated_with` |
| `one_hop_readout` | one group per readout relation: `associated_with`, `has_symptom`, `inverse:has_symptom`, `inverse:associated_with` |

`auxiliary_slots` must be below `batch_size` and divisible by the number of
groups (2 or 4). With `batch_size` 128 and `auxiliary_slots` 32, `two_hop` gives
96 one-hop, 16 forward and 16 inverse queries per update, and
`one_hop_readout` gives 96 one-hop queries and 8 per readout relation.

**Two-hop queries.** The index is built from `graph.npz` alone. Each anchor is
one query whose answers are the union over every training intermediate
disease, and negatives are filtered against that whole union. The subsampling
weight uses the same formula on the union size. Before training, the trainer
refuses to start unless the `associated_with` edges equal `training.tsv`, the
`has_symptom` edges equal the profiles of training diseases, every intermediate
is a training disease and none is a validation or test disease, the forward
pairs equal an independent join of `training.tsv` with those profiles, and the
inverse pairs are their transpose. The checks, the query and pair counts, and a
SHA-256 of each group's queries and answers are written under
`training_queries` in `identity.json`.

**Sampling.** Every group has its own sampler and random stream. The one-hop
group uses the default stream, so it draws the same queries, positives and
negatives in a `two_hop` and a `one_hop_readout` run with the same seed. Each
group wraps into its next epoch to fill its quota, so every update has exactly
`batch_size` queries; the default Track A sampler is unchanged and its last
batch of an epoch may still be short. The groups are encoded separately, their
logits are concatenated, and one KGReasoning loss with the source subsampling
weights covers all of them. The GeometrE membership term is computed once per
update. `training.jsonl` adds, for each group, its query loss and loss halves,
summed weight, share of the summed weight, and sampler epoch and offset.

**What the comparison can show.** The `one_hop_readout` control matches the
readout-relation share of each update, the one-hop queries and the budget. It
does not match the number of relation applications per query, the answer
distributions or the loss weights, so it does not isolate composition. Uniform
queries with a uniform positive per query do not give every gene–phenotype pair
the same probability, in either direction. Index construction draws no random
numbers and the model is built after seeding, as in Track A, so the step-0
parameters are expected to equal those of a Track A run with the same
configuration on the same hardware; this has not been verified. Compare tensors,
not checkpoint files, which differ because their identity and configuration
differ. Better training fit on two-hop paths
does not establish generalisation to new diseases, and a null result does not
show that composition is impossible.

## Running

```bash
python code/compositional_gda/train.py --config CONFIG.json --run-dir RUN_DIR \
    --device cuda --deadline-seconds 11400
```

| exit code | meaning |
|---|---|
| 0 | fixed budget and every scheduled validation complete; `COMPLETE.json` written |
| 75 | paused with a committed `resume.pt`; run the same command again to continue |
| 3 | another process holds the run-directory lock; nothing was done |
| other | failure; not resumable automatically |

The same command resumes a paused run. A completed run is verified and exits 0.
The trainer pauses with exit 75 when:

- the time since process start would exceed `--deadline-seconds` before the next
  update or validation finishes, estimated from the largest measured step,
  validation and commit times multiplied by `--deadline-safety` (default 1.5);
- it receives SIGUSR1 or SIGTERM, checked at every update boundary and between
  validation scoring chunks;
- `--stop-after N` is reached after update N, or `--stop-before-validation N`
  leaves the validation of update N pending.

A validation that is interrupted or skipped is replayed exactly once on resume.
Validation touches no random state or parameter, so the replay leaves the
training state unchanged.

## Outputs

| file | content |
|---|---|
| `identity.json` | configuration, protocol, data digests, trainer source digests, pinned references, package versions, device type and numeric settings |
| `resume.pt` | committed state: model, optimizer, scheduler (none), sampler position and generator (one per group for diagnostic runs), Python/NumPy/torch/CUDA RNG, validation curve, selections, step-0 baselines and artifact digests |
| `progress.json` | committed step, curve, selections and pending-validation flag |
| `training.jsonl` | one record per update (losses, gradient norm, GeometrE membership loss and its gradient norm, learning rate, sampler position, seconds, and per-group losses and weights for diagnostic runs) and per validation |
| `segments.jsonl` | start, pause and completion of each process, with host, Slurm job and GPU name |
| `checkpoints/step-NNNNNN.pt` | model weights at every validation step |
| `validation/step-NNNNNN.npz` | both readouts' phenotype-level scores, aggregated scores and raw and calibrated ranks, plus `scoring_query_chunk`, `scoring_score_chunk_elements`, `scoring_device` and `scoring_device_name` (GPU model, or `cpu`) of the process that scored it |
| `selected-{forward,inverse}.pt`, `selected-{forward,inverse}-validation.npz` | published selection for each readout |
| `COMPLETE.json` | final selections and SHA-256 of every retained artifact |

Elapsed time, host, job and GPU name are logged but excluded from the identity.
Changing the configuration, data, trainer source, package versions or device
type makes the trainer refuse the run directory. The chunk sizes are part of the
configuration and cannot change within a run, but the GPU model can change
between segments, so different validation steps of one run may come from
different GPU models; each archive names its own.

BetaE validation scores depend on `query_chunk` and on the device model: its
projection MLP uses matrix kernels whose results depend on the batch shape, and
the last query chunk is always smaller. On CPU at dimension 100 the difference
reached 1.5e-5. Re-scoring a saved checkpoint with other chunk sizes, on CPU, or
on another GPU model can therefore flip near-ties. TransE, Query2Box and
GeometrE scores were bit-identical across chunk sizes on CPU. Bit-identical
training updates across GPU models are not guaranteed either, so a run resumed on
another model is valid, but the resume tests do not show that it reproduces an
uninterrupted run bit for bit. Always
recompute metrics and selections from the saved phenotype-level scores.

## Recomputing selection

```bash
python code/compositional_gda/recompute.py --run-dir RUN_DIR --data-dir PREPARED_FOLD
```

This recomputes the aggregated scores, ranks, metrics and both selections from
the saved phenotype-level arrays and fails on any difference from the recorded
values. Selection replay refuses a curve that is empty, does not start at step 0,
or whose steps are not strictly increasing integers, because the first entry is
the baseline and the fallback.

## Training-fit diagnostic

```bash
python code/compositional_gda/fit_diagnostics.py --run-dir RUN_DIR --data-dir PREPARED_FOLD --output NEW_DIR \
    --device cuda --queries-per-relation 256 --seed 0 --expect-sample-sha256 SAMPLE_SHA256
```

This scores a completed run's checkpoint (by default the final step; or
`--checkpoint selected-forward.pt` and the like) on a fixed sample of
training-graph queries. It measures training fit, not validation or
generalisation, and scores no validation or test row, panel or profile. Six
structures are sampled: the four readout relations as one-hop queries and the
forward and inverse two-hop paths, each against a typed candidate set (training
diseases, has_symptom phenotypes or the frozen candidate genes). Two-hop answers
are the union over every training intermediate disease. For each structure the
script draws up to `--queries-per-relation` anchors without replacement and one
uniform positive per anchor; the sample depends only on the prepared fold, the
seed and the count, and `--expect-sample-sha256` refuses any other sample, so
every model is scored on the same queries. Each candidate is scored with
`score(..., negative=True)`, the validation form, and the positive is ranked
among the candidates that are not another known answer, with average-rank ties
and no calibration.

The run directory is read only. The script requires `COMPLETE.json` and checks
the identity hash, the checkpoint digest, keys, step, identity and
configuration, the prepared-data digests and the `models.py` digest the run
recorded; other trainer modules may differ and are listed in the output. It
writes `fit.npz` (sample, ranks and positive scores), `scores.npz` (the full
score matrices, unless `--no-score-matrices`), `summary.json`,
`provenance.json` and `fit.log` into `<output>.partial`, renamed to `<output>`
on success.

With 256 queries the binomial standard error of a Hits@k estimate is at most
about 0.031. That is a rough scale, not a significance threshold; every model is
scored on the same queries, so compare models with paired tests on the saved
per-query ranks.

## Tests

```bash
python -m pytest tests/test_compositional_models.py tests/test_compositional_resume.py
```

The model tests import the pinned KGReasoning and GeometrE sources (default
`~/Git/KGReasoning` and `~/Git/geometrE`; override with
`COMPOSITIONAL_KGREASONING` and `COMPOSITIONAL_GEOMETRE`). They check identical
initialisation for the same seed, identical one-hop and two-hop query
embeddings and scores, and identical training losses. The tests are skipped
when a source checkout is absent and fail when it is at another commit. The
resume tests run every variant on a toy fold and require interrupted runs to
reproduce an uninterrupted run bit for bit on the same device. They use the CPU
by default. Set `COMPOSITIONAL_TEST_DEVICE=cuda` to train and validate on a GPU
instead; the tests then fail, never skip, when CUDA is unavailable. Run the GPU
check on a GPU node, with a fresh `--basetemp` outside node-local `/tmp` so the
run directories survive (pytest empties an existing `--basetemp`):

```bash
COMPOSITIONAL_TEST_DEVICE=cuda CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest -v \
    "tests/test_compositional_resume.py::test_interrupted_runs_reproduce_uninterrupted_run" \
    --basetemp=OUT/basetemp --junitxml=OUT/junit.xml
```
