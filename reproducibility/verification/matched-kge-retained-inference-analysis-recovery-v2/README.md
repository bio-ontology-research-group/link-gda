# Matched retained-inference analysis recovery v2

The original CPU verifier job 54837466 failed after complete input/output hashing,
when its generic evidence helper tried to decode `stage-inputs.py` as JSON.
The failed native receipt, traceback and original plan are preserved here.
No numerical comparison had run and no full-equivalence proof was produced.

This version explicitly retains Python source by checksum without JSON decoding.
JSON evidence still undergoes strict decoding and all checksum/symlink guards
remain. All 53 preparation/regression tests pass, including wrong file hashes,
malformed JSON and retention of the exact triple-quoted Python input that caused
the failure. Run `python -m pytest test_preparation.py test_h200.py` here.

The numerical analysis suffix is byte-identical to the original verifier in
`../matched-kge-retained-inference-operation/analysis/`. All 843 input hashes,
400 fresh score comparisons, ten contrasts, 40 by 42 bias conditions, 5,600 rank
arrays and primary/bias exports remain required in a new native run. The plan
binds the actual completed inference capture and frozen reference identities.
The sbatch file uses the existing pinned Python 3.13/NumPy 2.2.6 environment.

Preparation tests establish the repair mechanism, not production equivalence.
Native completion and the full analysis receipt are still pending. The standard
portable scientific workflows remain documented under `../../../docs/`.
