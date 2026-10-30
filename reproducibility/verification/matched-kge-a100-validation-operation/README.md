# A100 validation replay completion

Native inference job 54764788 completed with exit zero on an A100 with eight
allocated CPUs and eight software threads. Fixed fold-zero U1 selections on
both graphs and T1 selections on the original graph reproduce all 144 recorded
raw/calibrated validation metrics for twelve selected checkpoints.

CPU step 54825093.132 rehashed the twelve retained A100 validation score files
(306,652,582 bytes) and completed with exit zero. Their hashes agree with the
separately completed [H200 validation replay](../matched-kge-h200-validation-inference.json).
The checkpoint hashes, selected epochs, candidate/profile counts and all metrics
also agree between these two fresh replays. Source, artifact and plan digests
bind the original 80-checkpoint interim audit and its unchanged scientific loop.

These are validation replays with audited native weights. Original training-time
validation score matrices were not retained, so agreement to training is checked
through the recorded aggregate metrics. Full test inference from independently
downloaded public weights remains outstanding. Follow the
[checkpoint workflow](../../../docs/matched-kge-checkpoints.md) for that stage.

Retained native paths identify this execution; the operation files are evidence
for the completed jobs. The portable workflow belongs to the linked guide.
