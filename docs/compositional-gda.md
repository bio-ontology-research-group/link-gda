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
| `resume.pt` | committed state: model, optimizer, scheduler (none), sampler position and generator, Python/NumPy/torch/CUDA RNG, validation curve, selections, step-0 baselines and artifact digests |
| `progress.json` | committed step, curve, selections and pending-validation flag |
| `training.jsonl` | one record per update (losses, gradient norm, GeometrE membership loss and its gradient norm, learning rate, sampler position, seconds) and per validation |
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
