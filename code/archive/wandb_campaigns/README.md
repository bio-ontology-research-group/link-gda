# Archived W&B campaign launchers

These scripts register or launch historical campaign groups and were archived
unchanged on 29 September 2026:

- run_sweep.sh: site-specific Slurm/W&B agent launcher.
- submit_sweeps.sh: submits the old registry groups with a 36-task default array.
- wandb_scripts/create_sweeps.py: registers a hardcoded collection of old sweep
  definitions. It is not the launch recipe for the current validation-selected runs.

The sweep YAML files and wandb_scripts/sweep_ids.yaml remain in their original
locations for provenance and for the retained W&B result readers.

These archived launchers retain historical path assumptions. create_sweeps.py
resolves its root from its own location; submit_sweeps.sh expects the historical
launcher and registry layout in its working directory. They are not runnable
from this archive without reconstructing that layout and its configuration.
No sweeps were registered or submitted during archiving.
