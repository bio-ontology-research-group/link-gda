# KGE execution and checkpoint provenance

The retained original-graph and no-bridge results are descriptive configurations,
not a matched estimate of disease-bridge removal. They differ in selection rule,
training budget, INDIGENA dimension and T1 positive filtering. Fixing execution
does not retroactively change those saved experiments.

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
