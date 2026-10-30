# KGE execution and checkpoint provenance

The [pinned matched-KGE environment](../reproducibility/environments/matched-kge/README.md)
provides the original Pixi manifest, lockfile and installed-package capture,
with a package/runtime verification command. It is separate from the ULTRA
training environment and the CPU saved-score analysis environment.

The retained original-graph and no-bridge results are descriptive configurations,
not a matched estimate of disease-bridge removal. They differ in selection rule,
training budget, INDIGENA dimension and T1 positive filtering. Fixing execution
does not retroactively change those saved experiments.

## Matched rerun

`code/training/run_matched_kge.py` supplies a separate matched original/no-bridge
workflow. Its [fixed protocol](../reproducibility/protocols/matched-kge-20261029.md)
uses dimension 400 for INDIGENA, dimension 200 for the other families, the same
filtered T1 sampler on both graphs, and the same epoch/selection rules within
each family. Training, global freeze and scoring are separate commands.
The [CPU integration](../reproducibility/verification/matched-kge-cpu.json) and
[full-graph GPU pilot](../reproducibility/verification/matched-kge-gpu-pilot.json)
pass for both graphs and all five families. The GPU pilot trained two epochs on
fold 0 and checked validation selections and warm-start provenance; it did not
score the test set. The production budget is specified in the protocol.

```bash
python code/training/run_matched_kge.py --phase train \
  --root "$MATCHED_RUN" --data-root "$PREPARED_DATA" \
  --fold 0 --graph original --model U1
```

Train U1 before ConvKB or P-pool on each graph/fold. INDIGENA additionally needs
`--gene-phenotypes` pointing to the frozen gene phenotype table. Install its
[separate frozen input supplement](frozen-inputs.md#matched-kge-annotation-supplement);
the installed path is `$WORKSPACE/inputs/matched-kge/inputs/gene_phenotypes.csv`.
Supply the same file for training and scoring. After all five
families on both graphs and all ten folds finish:

```bash
python code/training/run_matched_kge.py --phase freeze --root "$MATCHED_RUN"
python code/training/run_matched_kge.py --phase score \
  --root "$MATCHED_RUN" --data-root "$PREPARED_DATA" \
  --fold 0 --graph original --model U1
```

Repeat scoring for every planned model/graph/fold. The
[shared inference launcher](shared-kge-training.md#shared-inference-after-global-freeze)
schedules these same commands concurrently after a complete global freeze.
Matching epoch budgets does not equate gradient-update counts when graph sizes differ; the protocol states
this limit. Historical no-bridge outputs below remain separate artifacts.

After all 400 selected-checkpoint score files exist, build the manifest for the
shared evaluator. Store it directly inside the run root so the same bytes and
relative paths can be used in the public score archive:

```bash
export MATCHED_MANIFEST="$MATCHED_RUN/score-manifest.json"
python reproducibility/build_matched_kge_manifest.py \
  --root "$MATCHED_RUN" --data-root "$PREPARED_DATA" \
  --gene-phenotypes "$WORKSPACE/inputs/matched-kge/inputs/gene_phenotypes.csv" \
  --output "$MATCHED_MANIFEST"
python code/analysis/evaluate_stepwise_bias.py \
  --base-dir "$BENCHMARK_WORKSPACE" --score-manifest "$MATCHED_MANIFEST" \
  --out-json "$MATCHED_BIAS_REPORT" --ranks-dir "$MATCHED_RANKS" --workers 4
```

`$BENCHMARK_WORKSPACE/data` must be the same directory as `$PREPARED_DATA`.
The builder requires all 100 training runs and four choices per trajectory.
It checks the global freeze, production settings, scientific input identities,
source/environment agreement, prescribed U1 warm starts and every checkpoint
and score checksum. It independently recomputes the four strict validation
choices from every scheduled evaluation, including the final epoch; ties must
retain the earlier checkpoint. Output paths are relative to the manifest.
The evaluator then checks score axes and finite values and computes the 42
scoring/cohort conditions for all 40 model/graph/checkpoint combinations.

The manifest records the protocol's primary choice and scoring rule for each
family. Other validation choices remain sensitivities. Do not choose a new
primary checkpoint or scoring rule from the test report. The separate
[analysis specification](../reproducibility/protocols/matched-kge-analysis-20261029.md)
fixes ten primary contrasts and their paired bootstrap intervals before
production test scoring. Produce that summary and its tables with:

```bash
python code/analysis/summarize_matched_kge.py \
  --score-manifest "$MATCHED_MANIFEST" --report-json "$MATCHED_BIAS_REPORT" \
  --ranks-dir "$MATCHED_RANKS" --data-root "$PREPARED_DATA" \
  --output "$MATCHED_SUMMARY"
python reproducibility/generate_matched_kge_tables.py \
  --report-json "$MATCHED_SUMMARY" --out-dir "$MATCHED_TABLES"
```

The summary reopens and hashes all 100 primary score files, recomputes their
ranks, and checks both the saved ranks and primary aggregate statistics. It
also checks every rank-row identity and phenotype profile against the frozen
benchmark inputs. The tables show the ten prespecified contrasts and actual
graph sizes, epoch budgets and selected epochs. Other validation choices and
scoring/cohort conditions remain available in the full shared-evaluator report.
Final production evaluation and manuscript integration are still pending.

The [saved-score archive workflow](matched-kge-score-replay.md) packages all 400
scores with benchmark and selection evidence. Before export it recomputes every
evaluation condition, saved rank and primary comparison; a fresh extraction can
repeat this CPU analysis without model weights.

The [checkpoint archive workflow](matched-kge-checkpoints.md) preserves all
400 selected models and their complete validation evidence. It reuses the
verified campaign manifest, checks the packed inventory, and provides the
original trainer commands for inference after extraction into a new workspace.

## No-bridge driver

`code/training/retrain_nobridge_kge.py` reads `DATA/foldN/{raw/train.txt,
candidates.txt,profiles.json,training.tsv,validation.tsv,test.tsv}`. The combined
U1/INDIGENA/P-pool condition also reads `data/gene_phenotypes.csv` relative to
`--code-root`. Run it on prepared inputs, with a fresh output root:

```bash
python code/training/retrain_nobridge_kge.py \
  --data-root "$PREPARED_DATA" --code-root "$PWD" \
  --out-root "$NEW_OUTPUT" --fold 0 --condition U1_INDIGENA_PPOOL
python code/training/retrain_nobridge_kge.py \
  --data-root "$PREPARED_DATA" --code-root "$PWD" \
  --out-root "$NEW_OUTPUT" --fold 0 --condition T1
python code/training/retrain_nobridge_kge.py \
  --data-root "$PREPARED_DATA" --code-root "$PWD" \
  --out-root "$NEW_OUTPUT" --fold 0 --condition CONVKB
```

The default candidate count is 13,625. `--expected-candidates` permits explicit
small integration fixtures; changing it is not the paper's benchmark. All input
split identities and candidate membership are checked. Both `associated_with`
and `has_symptom` are removed before training; explicit disease identifiers must
be absent from the resulting graph.

Run the same command again to resume. Epoch snapshots jointly preserve model
weights, Adam state, Python/NumPy/PyTorch random states, validation history,
selection criteria and selected checkpoint files. Profile snapshots additionally
preserve the pooling parameter and the ranker's independent NumPy generator.
Atomic generation commits prevent a later, uncommitted validation checkpoint
from changing a resumed selection. Source, input and configuration changes are
rejected. Completed score/checkpoint artifacts are checksum-checked before skip.
Historical selected weights without full state cannot be used for resumption.

PyKEEN 1.11.1 is required. Automatic batch-size probing is disabled; the requested
batch and subbatch sizes are part of resume identity. `--epochs` is the total
budget, including completed epochs. ConvKB now respects that budget instead of
silently capping it at 100, with the retained 15-validation-evaluation patience
on calibrated mean rank. This produces a new experiment, not a replay of the
old capped run. Technical `--stop-after-epoch` and `--stop-after-profile-epoch`
options stop without exporting incomplete scores.

The combined condition still fits U1 and INDIGENA on the same dimension-200
TransD trajectory, and uses the historical unfiltered T1 sampler. Original-graph
INDIGENA used dimension 400; the original top-hit T1 campaign used filtered
sampling. A scientifically matched graph comparison must equalize these choices
and selection rules. This driver alone does not supply that comparison.

## Original-graph top-hit configuration entry point

The retained campaign uses `kge_transd.py --head_experiment CONFIG.json`, not the
previously documented nonexistent `kge_head_experiment.py`. A graph-phase
configuration has this form (choose a new output directory):

```json
{
  "phase": "graph",
  "condition": "T1",
  "fold": 0,
  "seed": 0,
  "epochs": 1000,
  "validation_frequency": 20,
  "sub_batch_size": 8192,
  "output": "results/top-hits/T1-fold0"
}
```

```bash
python code/training/kge_transd.py --fold 0 \
  --use_phenotypes --use_functions --use_site --use_graph \
  --projector_name owl2vecstar --graph_policy independent-unique \
  --candidate_pool phenotype --calibration_unit disease \
  --random_seed 0 --val_seed 0 --embedding_dim 200 \
  --batch_size 65536 --learning_rate 0.001 --no_sweep \
  --head_experiment CONFIG.json
```

This route excludes validation and test disease nodes before training and uses
`link_gda.filtered_negative_sampling` for T1/T32, preserving the retained top-hit
campaign's rejection of known positives. The older generic trainer and no-bridge
driver retain `link_gda.negative_sampling`; those modules are not interchangeable.
Graph training, profile training and frozen-selection test scoring are separate
phases in `head_experiment.py`. This historical route still requires a fresh
output directory; the new resume support above applies to the no-bridge driver.

## Verification scope

`tests/test_training_resume.py` trains real small TransD graphs and a real
end-to-end profile ranker. Uninterrupted and interrupted/resumed trajectories
must agree bitwise in weights, optimizer, random state, losses and selection.
`tests/test_nobridge_execution.py` runs the real T1 CLI through training,
interruption, resumption and score export, then checks rejection of a corrupted
completed score file. These are CPU regression checks, not a full-data GPU rerun
or proof that the historical KGE comparisons were matched.
