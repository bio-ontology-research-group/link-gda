# CLMPT suite: ComplEx pretraining

ComplEx (rank 400, N3, sparse Adagrad, batch 1000) on a modality graph with a configurable seed
(0 to 9) and exact full-state resume. The objective, initialization and batch-order algorithm are
those of `code/clmpt_boundary/train.py`; seed 0 follows that trainer's initialization and order on
the same graph. No result in this repository has been produced by this package yet.

```
python make_config.py --protocol protocols/suite-pretrain-v1.json --purpose qualification --experiment-id ID \
  --data-dir GRAPH_DIR --recipe-id plr0.4-n3r0.05-u633450 --seed 1 --maximum-steps 40 --output run.json
python train.py --config run.json --output RUN_DIR [--allocation-stop-at UNIXTIME]
```

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=0`. Return codes: 0 complete, 75 paused
(rerun to resume), otherwise error with `ATTENTION.json`. Each full checkpoint stores the model,
Adagrad state, epoch, current triple permutation and position, and all generator states. The
model-only initializer is written to `models/update-<total>.pt`.

Protocol v1 admits qualification only. Production pretraining requires a later protocol that
names the frozen recipe and the admitted modality graphs; every graph then trains for the same
number of optimizer updates.
