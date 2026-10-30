# Independent matched inference from retained weights

This operation replaces the stopped GitHub bulk-download route. It preserves
all ten folds, 100 matched training runs and 400 validation-selected weights.
The frozen inputs, model source, scoring options and numerical comparisons are
unchanged. It does not establish a new public download or independent retraining.

ORIX array 356481 has three allocations, each with one H200, eight CPUs and
eight software threads. Task 0 scores folds 0/3/6/9, task 1 scores 1/4/7 and
task 2 scores 2/5/8. This fits the account's submission limit without cancelling
unrelated jobs. All ten fold completion records and all 400 new score files are
required before capture can pass.

`stage-inputs.py` links exactly 843 retained members into a fresh workspace and
excludes original predictions. Its preparation receipt checks metadata only.
`run-inference.py` hashes every assigned member before and after scoring: 87
members and 40 weights per fold, covering all 843 members across the campaign.
The original scorer, strict model loads, source identities, freeze, validation
selections, runtime and output checks are retained. The scorer's 100 empty lock
files are explicit operational metadata; the completed workspace has 1,443
files (843 inputs, 400 scores, 100 scorer records, 100 empty locks).

ORIX changes mtime during hard-link creation. The helper checks inode/device,
size and same-file identity during linking; mtime is recorded rather than used
as content evidence. All mandatory SHA256 checks remain. A native small-file
probe and a regression test cover this behavior. Failed preparation and rejected
submission attempts remain in the local audit; no GPU inference ran in those
attempts.

The source proof binds public analysis commit 68101ed and original training
commit 063efc2. The H200 validation reference binds the completed twelve-weight,
144-metric replay. `source-plan.json` and `prepare-source.py` reconstruct the
small source checkouts; no model/data archives are fetched from GitHub.

For IBEX, `prepare-analysis-inputs.py` links and hashes all retained input members
on allocated CPUs. Its recovery plan uses canonical `/ibex/project` paths,
because the `/ibex/scratch` aliases are symlinks. The failed preflight created no
workspace. Job 54832268 is the versioned recovery of failed job 54832169.

After successful GPU completion, run `capture-native.py --array 356481`, use the
separate retained-output relay, and run the full retained-inference CPU verifier.
The relay sends only fresh predictions and small metadata; the weight files
remain on their servers. CPU comparison must verify all 400 predictions, ten
primary contrasts, 40 variants by 42 bias conditions, 5,600 rank arrays and all
primary/bias exports. Preparation tests and running jobs do not prove those
scientific comparisons.
