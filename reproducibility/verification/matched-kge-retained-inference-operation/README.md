# Retained-input inference audit

These small source and metadata files record the author-side independent
inference check after withdrawal of temporary GitHub bulk attachments. They
contain no model weights or prediction arrays. The original training source,
scorer settings, selected weights and complete scientific comparisons remain
unchanged. See [current evidence](../matched-kge-retained-inference.json).

The `inference`, `analysis` and `transfer` directories retain actual execution
scripts and their configuration/provenance. The scripts record the original
cluster paths and native validation reference; they are not a replacement for
the [portable paper workflows](../../../docs/reproduction.md). Running these
author-side execution scripts requires access to the retained data. The final
replacement public research archive is still outstanding.

All CPU preparation tests can run without the large data files. Their fixtures
contain only frozen file identities, scorer metadata and earlier verifier source.
From the repository root, use the [test environment](../../../docs/testing.md):

```bash
PYTHONPATH=code:tests python3 -m pytest -q \
  reproducibility/verification/matched-kge-retained-inference-operation/inference \
  reproducibility/verification/matched-kge-retained-inference-operation/analysis \
  reproducibility/verification/matched-kge-retained-inference-operation/transfer
```

Passing preparation tests, input hashes or live inference jobs does not establish
complete numerical agreement. Require actual native completion, fresh prediction
transfer and all downstream score/rank/contrast/bias/export comparisons.
