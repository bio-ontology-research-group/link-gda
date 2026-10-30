# Retained-input inference output relay

Prepared for complete ORIX array 356481, not yet executed. Require actual native
capture after all three allocations and all ten folds complete. Require the
successful IBEX `RETAINED_INPUTS_STAGED.json` before metadata staging.

The two-stream relay copies only the 400 fresh prediction arrays and small
captured metadata. It never stages their bytes on WS and does not transfer model
weights. The closed destination workspace is 1,443 files: 843 inputs, 400 score
arrays, 100 scorer records and 100 empty operational locks. All lock bytes and
metadata are checked; full input/output hashes and numerical equivalence are
the responsibility of the separate allocated CPU verifier.

After actual capture, populate `plan.json` with its SHA256, the actual inference
plan/array, the successful input-stage proof, source/destination roots and all
three helper hashes. Stage metadata on IBEX, run the relay on WS, then require
`DESTINATION_READY.json` and `OUTPUT_TRANSFER_COMPLETE.json` before analysis.
